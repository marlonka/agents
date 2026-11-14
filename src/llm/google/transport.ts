import { createParser } from 'eventsource-parser';
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
import type { EventSourceParser, EventSourceMessage } from 'eventsource-parser';

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
  let parser: EventSourceParser;
  let endsWithCR = false;
  let atEnd = false;
  const events = body
    .pipeThrough(new TextDecoderStream('utf-8', { fatal: true }))
    .pipeThrough(
      new TransformStream<string, EventSourceMessage>({
        start(controller): void {
          parser = createParser({
            onEvent(event): void {
              if (atEnd) {
                throw new GoogleGenerativeAIError(
                  'Incomplete Gemini stream event'
                );
              }
              controller.enqueue(event);
            },
          });
        },
        transform(chunk): void {
          endsWithCR = chunk.endsWith('\r');
          parser.feed(chunk);
        },
        flush(): void {
          // Resolve a trailing CR, which the parser holds until it can check for LF.
          if (endsWithCR) {
            parser.feed('\n');
          }
          // Reject any data without a terminating blank line.
          atEnd = true;
          parser.feed('\n\n');
        },
      })
    );
  let receivedData = false;

  for await (const event of events) {
    if (event.event === 'error') {
      throw new GoogleGenerativeAIError(
        'Gemini gateway returned a stream error'
      );
    }
    const data = event.data.trim();
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
