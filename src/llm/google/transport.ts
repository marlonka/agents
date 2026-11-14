import {
  getLines,
  getMessages,
} from '@langchain/core/utils/event_source_parse';
import {
  GoogleGenerativeAIError,
  GoogleGenerativeAIFetchError,
  GoogleGenerativeAIRequestInputError,
} from '@google/generative-ai';
import type {
  GenerateContentResponse,
  GenerateContentRequest,
  SingleRequestOptions,
  GenerativeModel,
  RequestOptions,
  ErrorDetails,
} from '@google/generative-ai';
import type { EventSourceMessage } from '@langchain/core/utils/event_source_parse';

type GoogleErrorResponse = {
  error?: { code?: number; message?: string; details?: ErrorDetails[] };
};

type StreamResult = { stream: AsyncGenerator<GenerateContentResponse> };

export type GoogleTransport = Omit<
  GenerativeModel,
  'cachedContent' | 'generateContentStream'
> & {
  cachedContent?: GenerativeModel['cachedContent'];
  generateContentStream(
    request: GenerateContentRequest,
    options?: SingleRequestOptions
  ): Promise<StreamResult>;
};

async function* readGoogleStream(
  body: ReadableStream<Uint8Array>
): AsyncGenerator<GenerateContentResponse> {
  let parse: ReturnType<typeof getLines>;
  const events = new TransformStream<Uint8Array, EventSourceMessage>({
    start(controller): void {
      parse = getLines(getMessages((event) => controller.enqueue(event)));
    },
    transform(chunk): void {
      parse(chunk);
    },
  });
  const reader = body.pipeThrough(events).getReader();
  let receivedData = false;

  try {
    for (
      let read = await reader.read();
      !read.done;
      read = await reader.read()
    ) {
      const { value } = read;
      if (value.event === 'error') {
        throw new GoogleGenerativeAIError(
          'Gemini gateway returned a stream error'
        );
      }
      const data = value.data.trim();
      if (data === '[DONE]') {
        break;
      }
      if (data === '') {
        continue;
      }

      let response: (GenerateContentResponse & GoogleErrorResponse) | null;
      try {
        response = JSON.parse(data);
      } catch {
        throw new GoogleGenerativeAIError('Invalid JSON in Gemini stream');
      }
      if (
        response == null ||
        typeof response !== 'object' ||
        Array.isArray(response)
      ) {
        throw new GoogleGenerativeAIError('Invalid response in Gemini stream');
      }
      if (response.error != null) {
        throw new GoogleGenerativeAIFetchError(
          'Gemini gateway returned a stream error',
          typeof response.error.code === 'number'
            ? response.error.code
            : undefined
        );
      }
      receivedData = true;
      yield response;
    }
    if (!receivedData) {
      throw new GoogleGenerativeAIError('Gemini stream ended without data');
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

/** Route both LangChain streaming APIs through one reader with no aggregation branch. */
export function attachGoogleStreaming(
  client: GoogleTransport,
  defaults: RequestOptions = {}
): void {
  client.generateContentStream = async (
    request,
    options
  ): Promise<StreamResult> => {
    const { apiClient, apiVersion, baseUrl, customHeaders, signal, timeout } = {
      ...defaults,
      ...options,
    };
    const timeoutSignal =
      timeout != null && timeout >= 0
        ? AbortSignal.timeout(timeout)
        : undefined;
    const requestSignal =
      signal && timeoutSignal
        ? AbortSignal.any([signal, timeoutSignal])
        : (signal ?? timeoutSignal);
    requestSignal?.throwIfAborted();

    const headers = new Headers(customHeaders);
    if (headers.has('x-goog-api-key') || headers.has('x-goog-api-client')) {
      throw new GoogleGenerativeAIRequestInputError(
        'Cannot set reserved Google API headers'
      );
    }
    headers.set('content-type', 'application/json');
    headers.set('x-goog-api-key', client.apiKey);
    if (apiClient != null && apiClient !== '') {
      headers.set('x-goog-api-client', apiClient);
    }

    const base =
      baseUrl == null || baseUrl === ''
        ? 'https://generativelanguage.googleapis.com'
        : baseUrl.replace(/\/+$/, '');
    const version =
      apiVersion != null && apiVersion !== '' ? apiVersion : 'v1beta';
    const url = `${base}/${version}/${client.model}:streamGenerateContent?alt=sse`;
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        generationConfig: client.generationConfig,
        safetySettings: client.safetySettings,
        tools: client.tools,
        toolConfig: client.toolConfig,
        systemInstruction: client.systemInstruction,
        cachedContent: client.cachedContent?.name,
        ...request,
      }),
      signal: requestSignal,
    });

    if (!response.ok) {
      const failure: GoogleErrorResponse | null = await response
        .json()
        .catch(() => null);
      requestSignal?.throwIfAborted();
      const { status, statusText } = response;
      const message = failure?.error?.message;
      throw new GoogleGenerativeAIFetchError(
        `[${status} ${statusText}]${typeof message === 'string' ? ` ${message}` : ''}`,
        status,
        statusText,
        failure?.error?.details
      );
    }
    if (response.body == null) {
      throw new GoogleGenerativeAIError('Gemini stream response has no body');
    }
    return { stream: readGoogleStream(response.body) };
  };
}
