import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { getEnvironmentVariable } from '@langchain/core/utils/env';
import {
  FunctionCallingMode,
  GoogleGenerativeAI as GenerativeAI,
} from '@google/generative-ai';
import type {
  GenerateContentRequest,
  RequestOptions,
  CachedContent,
  SafetySetting,
  ModelParams,
  ToolConfig,
} from '@google/generative-ai';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type { ChatResult } from '@langchain/core/outputs';
import type { GeminiApiUsageMetadata, InputTokenDetails } from './types';
import type { GoogleClientOptions, GoogleThinkingConfig } from '@/types';
import type { GoogleTransport } from './transport';
import {
  convertResponseContentToChatGenerationChunk,
  convertBaseMessagesToContent,
  isGemini3PlusModel,
  dropUnsupportedModelTurnPrefill,
  mapGenerateContentResultToChatResult,
} from './utils/common';
import { smoothGenerationChunks } from '@/llm/stream/chunkAdapters';
import { resolveStreamDelay } from '@/llm/stream/smoother';
import { attachGoogleStreaming } from './transport';

type GoogleToolConfigWithServerSideInvocations = ToolConfig & {
  includeServerSideToolInvocations?: boolean;
  functionCallingConfig?: Omit<
    NonNullable<ToolConfig['functionCallingConfig']>,
    'mode'
  > & {
    mode?:
      | NonNullable<ToolConfig['functionCallingConfig']>['mode']
      | 'VALIDATED';
  };
};

export class CustomChatGoogleGenerativeAI extends ChatGoogleGenerativeAI {
  _lc_stream_delay: number;
  thinkingConfig?: GoogleThinkingConfig;
  includeServerSideToolInvocations?: boolean;

  /**
   * Override to add gemini-3 model support for multimodal and function calling thought signatures
   */
  /** Gemini 3 and later accept image, audio, video and PDF input. */
  get _isMultimodalModel(): boolean {
    return (
      this.model.startsWith('gemini-1.5') ||
      this.model.startsWith('gemini-2') ||
      (this.model.startsWith('gemma-3-') &&
        !this.model.startsWith('gemma-3-1b')) ||
      isGemini3PlusModel(this.model)
    );
  }

  constructor(fields: GoogleClientOptions) {
    super(fields);

    this._lc_stream_delay = resolveStreamDelay(fields._lc_stream_delay);
    this.model = fields.model.replace(/^models\//, '');

    this.maxOutputTokens = fields.maxOutputTokens ?? this.maxOutputTokens;

    if (this.maxOutputTokens != null && this.maxOutputTokens < 0) {
      throw new Error('`maxOutputTokens` must be a positive integer');
    }

    this.temperature = fields.temperature ?? this.temperature;
    if (
      this.temperature != null &&
      (this.temperature < 0 || this.temperature > 2)
    ) {
      throw new Error('`temperature` must be in the range of [0.0,2.0]');
    }

    this.topP = fields.topP ?? this.topP;
    if (this.topP != null && this.topP < 0) {
      throw new Error('`topP` must be a positive integer');
    }

    if (this.topP != null && this.topP > 1) {
      throw new Error('`topP` must be below 1.');
    }

    this.topK = fields.topK ?? this.topK;
    if (this.topK != null && this.topK < 0) {
      throw new Error('`topK` must be a positive integer');
    }

    this.stopSequences = fields.stopSequences ?? this.stopSequences;

    this.apiKey = fields.apiKey ?? getEnvironmentVariable('GOOGLE_API_KEY');
    if (this.apiKey == null || this.apiKey === '') {
      throw new Error(
        'Please set an API key for Google GenerativeAI ' +
          'in the environment variable GOOGLE_API_KEY ' +
          'or in the `apiKey` field of the ' +
          'ChatGoogleGenerativeAI constructor'
      );
    }

    this.safetySettings = fields.safetySettings ?? this.safetySettings;
    if (this.safetySettings && this.safetySettings.length > 0) {
      const safetySettingsSet = new Set(
        this.safetySettings.map((s) => s.category)
      );
      if (safetySettingsSet.size !== this.safetySettings.length) {
        throw new Error(
          'The categories in `safetySettings` array must be unique'
        );
      }
    }

    this.thinkingConfig = fields.thinkingConfig ?? this.thinkingConfig;
    this.includeServerSideToolInvocations =
      fields.includeServerSideToolInvocations ??
      this.includeServerSideToolInvocations;

    this.streaming = fields.streaming ?? this.streaming;
    this.json = fields.json;

    const requestOptions = {
      apiVersion: fields.apiVersion,
      baseUrl: fields.baseUrl,
      customHeaders: fields.customHeaders,
    };
    const client = new GenerativeAI(this.apiKey).getGenerativeModel(
      {
        model: this.model,
        safetySettings: this.safetySettings as SafetySetting[],
        generationConfig: {
          stopSequences: this.stopSequences,
          maxOutputTokens: this.maxOutputTokens,
          temperature: this.temperature,
          topP: this.topP,
          topK: this.topK,
          ...(this.json != null
            ? { responseMimeType: 'application/json' }
            : {}),
        },
      },
      requestOptions
    );
    attachGoogleStreaming(client, requestOptions);
    // @ts-expect-error LangChain's native streamEvents uses this private client.
    this.client = client;
    this.streamUsage = fields.streamUsage ?? this.streamUsage;
  }

  private get googleClient(): GoogleTransport {
    // @ts-expect-error Both streaming paths must use the same LangChain client.
    return this.client;
  }

  useCachedContent(
    cachedContent: CachedContent,
    modelParams?: ModelParams,
    requestOptions?: RequestOptions
  ): void {
    super.useCachedContent(cachedContent, modelParams, requestOptions);
    attachGoogleStreaming(this.googleClient, requestOptions);
  }

  static lc_name(): 'LibreChatGoogleGenerativeAI' {
    return 'LibreChatGoogleGenerativeAI';
  }

  /**
   * Helper function to convert Gemini API usage metadata to LangChain format
   * Includes support for cached tokens and tier-based tracking for gemini-3-pro-preview
   */
  private _convertToUsageMetadata(
    usageMetadata: Partial<GeminiApiUsageMetadata> | undefined,
    model: string
  ): UsageMetadata | undefined {
    if (!usageMetadata) {
      return undefined;
    }

    const output: UsageMetadata = {
      input_tokens: usageMetadata.promptTokenCount ?? 0,
      output_tokens:
        (usageMetadata.candidatesTokenCount ?? 0) +
        (usageMetadata.thoughtsTokenCount ?? 0),
      total_tokens: usageMetadata.totalTokenCount ?? 0,
    };

    const cachedTokens = usageMetadata.cachedContentTokenCount ?? 0;
    if (cachedTokens > 0) {
      output.input_token_details ??= {};
      output.input_token_details.cache_read = cachedTokens;
    }

    // gemini-3-pro-preview has bracket based tracking of tokens per request
    if (model === 'gemini-3-pro-preview') {
      const over200k = Math.max(
        0,
        (usageMetadata.promptTokenCount ?? 0) - 200000
      );
      const cachedOver200k = Math.max(
        0,
        (usageMetadata.cachedContentTokenCount ?? 0) - 200000
      );
      if (over200k) {
        output.input_token_details = {
          ...output.input_token_details,
          over_200k: over200k,
        } as InputTokenDetails;
      }
      if (cachedOver200k) {
        output.input_token_details = {
          ...output.input_token_details,
          cache_read_over_200k: cachedOver200k,
        } as InputTokenDetails;
      }
    }

    return output;
  }

  invocationParams(
    options?: this['ParsedCallOptions']
  ): Omit<GenerateContentRequest, 'contents'> {
    const params = super.invocationParams(options);
    if (
      this.includeServerSideToolInvocations === true &&
      Array.isArray(params.tools) &&
      params.tools.length > 0
    ) {
      const toolConfig = params.toolConfig as
        | GoogleToolConfigWithServerSideInvocations
        | undefined;
      const functionCallingConfig = toolConfig?.functionCallingConfig;
      params.toolConfig = {
        ...toolConfig,
        ...(functionCallingConfig?.mode === FunctionCallingMode.AUTO
          ? {
            functionCallingConfig: {
              ...functionCallingConfig,
              mode: 'VALIDATED',
            },
          }
          : {}),
        includeServerSideToolInvocations: true,
      } as ToolConfig;
    }
    return params;
  }

  private buildRequest(
    messages: BaseMessage[],
    options: this['ParsedCallOptions']
  ): GenerateContentRequest {
    const prompt =
      convertBaseMessagesToContent(
        messages,
        this._isMultimodalModel,
        this.useSystemInstruction,
        this.model
      ) ?? [];
    const systemInstruction =
      prompt[0]?.role === 'system' ? prompt[0] : undefined;
    return {
      ...this.invocationParams(options),
      ...(systemInstruction && { systemInstruction }),
      contents:
        dropUnsupportedModelTurnPrefill(
          systemInstruction ? prompt.slice(1) : prompt,
          this.model
        ) ?? [],
    };
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const { signal } = options;
    signal?.throwIfAborted();
    const request = this.buildRequest(messages, options);

    const res = await this.caller.callWithOptions({ signal }, async () => {
      signal?.throwIfAborted();
      try {
        return await this.googleClient.generateContent(request, { signal });
      } catch (error) {
        signal?.throwIfAborted();
        throw error;
      }
    });

    const response = res.response;
    const usageMetadata = this._convertToUsageMetadata(
      response.usageMetadata,
      this.model
    );

    const generationResult = mapGenerateContentResultToChatResult(response, {
      usageMetadata,
    });

    await runManager?.handleLLMNewToken(
      generationResult.generations[0]?.text ?? '',
      undefined,
      undefined,
      undefined,
      undefined,
      undefined
    );
    return generationResult;
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const controller = new AbortController();
    const signal = options.signal
      ? AbortSignal.any([options.signal, controller.signal])
      : controller.signal;
    try {
      yield* smoothGenerationChunks({
        chunks: this._streamProviderChunks(messages, options, signal),
        delayMs: this._lc_stream_delay,
        signal: options.signal,
        runManager,
        abortUpstream: () => controller.abort(),
      });
    } finally {
      controller.abort();
    }
  }

  private async *_streamProviderChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    signal: AbortSignal
  ): AsyncGenerator<ChatGenerationChunk> {
    signal.throwIfAborted();
    const request = this.buildRequest(messages, options);
    const { stream } = await this.caller.callWithOptions({ signal }, () =>
      this.googleClient.generateContentStream(request, { signal })
    );

    let index = 0;
    let lastUsageMetadata: UsageMetadata | undefined;
    for await (const response of stream) {
      if (
        'usageMetadata' in response &&
        this.streamUsage !== false &&
        options.streamUsage !== false
      ) {
        lastUsageMetadata = this._convertToUsageMetadata(
          response.usageMetadata,
          this.model
        );
      }

      const chunk = convertResponseContentToChatGenerationChunk(response, {
        usageMetadata: undefined,
        index,
      });
      index += 1;
      if (!chunk) {
        continue;
      }

      yield chunk;
    }

    if (lastUsageMetadata) {
      yield new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          usage_metadata: lastUsageMetadata,
        }),
      });
    }
  }
}
