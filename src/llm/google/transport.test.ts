import { once } from 'node:events';
import { createServer } from 'node:http';
import { describe, expect, it } from '@jest/globals';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import type { RequestListener, ServerResponse } from 'node:http';
import { CustomChatGoogleGenerativeAI } from './index';

const model = 'gemini-3-flash-preview';
const usage = {
  promptTokenCount: 5,
  candidatesTokenCount: 2,
  thoughtsTokenCount: 3,
  totalTokenCount: 10,
};

function event(text: string): string {
  return JSON.stringify({
    candidates: [{ content: { role: 'model', parts: [{ text }] }, index: 0 }],
  });
}

function openStream(response: ServerResponse): void {
  response.writeHead(200, { 'Content-Type': 'text/event-stream' });
}

async function withGateway(
  handler: RequestListener,
  run: (client: CustomChatGoogleGenerativeAI) => Promise<void>,
  streamDelay = 0
): Promise<void> {
  const gateway = createServer(handler);
  gateway.listen(0, '127.0.0.1');
  await once(gateway, 'listening');
  const address = gateway.address();
  if (address == null || typeof address === 'string') {
    throw new Error('Test gateway has no TCP address');
  }
  try {
    await run(
      new CustomChatGoogleGenerativeAI({
        model,
        apiKey: 'test-key',
        apiVersion: 'v1',
        baseUrl: `http://127.0.0.1:${address.port}/gateway/`,
        customHeaders: {
          Authorization: 'Bearer gateway-key',
          'X-Test': 'proxy',
        },
        thinkingConfig: { includeThoughts: true, thinkingLevel: 'HIGH' },
        maxRetries: 0,
        _lc_stream_delay: streamDelay,
      })
    );
  } finally {
    gateway.closeAllConnections();
    await new Promise<void>((resolve) => gateway.close(() => resolve()));
  }
}

async function expectDisconnected(disconnected: Promise<void>): Promise<void> {
  const timeout = Promise.withResolvers<never>();
  const timer = setTimeout(
    () => timeout.reject(new Error('Gateway response stayed open')),
    2000
  );
  try {
    await Promise.race([disconnected, timeout.promise]);
  } finally {
    clearTimeout(timer);
  }
}

describe('Google SSE gateway', () => {
  it('streams across heartbeats and stops at [DONE] while the gateway remains open', async () => {
    const disconnected = Promise.withResolvers<void>();
    let request: {
      url?: string;
      auth?: string;
      key?: string;
      header?: string;
      body?: object;
    } = {};
    await withGateway(
      async (req, res) => {
        req.setEncoding('utf8');
        let body = '';
        for await (const data of req) {
          body += data;
        }
        request = {
          url: req.url,
          auth: req.headers.authorization,
          key: req.headers['x-goog-api-key'] as string,
          header: req.headers['x-test'] as string,
          body: JSON.parse(body),
        };
        res.once('close', disconnected.resolve);
        openStream(res);
        res.write(`\n: initial keepalive\n\ndata:${event('before ')}\n\n`);
        const tail = Buffer.from(
          `\n: thinking\r\n\r\nid: 2\r\ndata: {\r\ndata: ${event('after 🦀').slice(1)}\r\n\r\n` +
            `data: ${JSON.stringify({ usageMetadata: usage })}\n\ndata: [DONE]\n\ndata: ignored\n\n`
        );
        const split = tail.indexOf(Buffer.from('🦀')) + 1;
        res.write(tail.subarray(0, split));
        setImmediate(() => {
          res.write(tail.subarray(split));
        });
      },
      async (client) => {
        const stream = await client.stream(
          [new SystemMessage('Be concise'), new HumanMessage('Hello')],
          { signal: AbortSignal.timeout(3000) }
        );
        const chunks = await Array.fromAsync(stream);
        expect(chunks.map((chunk) => chunk.text).join('')).toBe(
          'before after 🦀'
        );
        expect(chunks.at(-1)?.usage_metadata).toEqual({
          input_tokens: 5,
          output_tokens: 5,
          total_tokens: 10,
        });
        expect(request).toMatchObject({
          url: `/gateway/v1/models/${model}:streamGenerateContent?alt=sse`,
          auth: 'Bearer gateway-key',
          key: 'test-key',
          header: 'proxy',
          body: {
            contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
            systemInstruction: {
              role: 'system',
              parts: [{ text: 'Be concise' }],
            },
            generationConfig: {
              thinkingConfig: { includeThoughts: true, thinkingLevel: 'HIGH' },
            },
          },
        });
        await expectDisconnected(disconnected.promise);
      }
    );
  }, 5000);

  it('uses the same transport for typed stream events', async () => {
    await withGateway(
      (_req, res) => {
        openStream(res);
        res.end(
          `:\r\ndata: ${event('first ')}\r\n\r\n: keep-alive\r\ndata: ${event('second')}\r\n\r\n`
        );
      },
      async (client) => {
        const stream = client.streamEvents('Hello');
        await expect(Promise.resolve(stream.text)).resolves.toBe(
          'first second'
        );
      }
    );
  });

  it('consumes a complete CR-delimited event at EOF', async () => {
    await withGateway(
      (_req, res) => {
        openStream(res);
        res.end(`data: ${event('complete')}\r\r`);
      },
      async (client) => {
        const chunks = await Array.fromAsync(await client.stream('Hello'));
        expect(chunks.map((chunk) => chunk.text).join('')).toBe('complete');
      }
    );
  });

  it.each([
    ['chunk streaming', 'streamGenerateContent'],
    ['typed events', 'streamGenerateContent'],
    ['non-streaming', 'generateContent'],
  ])(
    'closes the upstream response on Stop during %s',
    async (mode, method) => {
      const disconnected = Promise.withResolvers<void>();
      const started = Promise.withResolvers<void>();
      await withGateway(
        (req, res) => {
          const streaming = method === 'streamGenerateContent';
          expect(req.url).toContain(`:${method}`);
          res.once('close', disconnected.resolve);
          if (streaming) {
            openStream(res);
            res.write(`data: ${event('first')}\n\n: thinking\n\n`);
          } else {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.write('{"candidates":[');
          }
          started.resolve();
        },
        async (client) => {
          const stop = new AbortController();
          if (mode === 'typed events') {
            const stream = client.streamEvents('Hello', {
              signal: stop.signal,
            });
            const iterator = stream.text[Symbol.asyncIterator]();
            expect((await iterator.next()).value).toBe('first');
            const pending = iterator.next();
            stop.abort();
            await expect(pending).rejects.toThrow(/abort/i);
          } else if (method === 'streamGenerateContent') {
            const stream = await client.stream('Hello', {
              signal: stop.signal,
            });
            const iterator = stream[Symbol.asyncIterator]();
            expect((await iterator.next()).value?.text).toBe('first');
            const pending = iterator.next();
            stop.abort();
            await expect(pending).rejects.toThrow(/abort/i);
          } else {
            const result = client.invoke('Hello', { signal: stop.signal });
            await started.promise;
            stop.abort();
            await expect(result).rejects.toThrow(/abort/i);
          }
          await expectDisconnected(disconnected.promise);
        },
        25
      );
    },
    5000
  );

  it('closes a smoothed stream when its consumer returns early', async () => {
    const disconnected = Promise.withResolvers<void>();
    await withGateway(
      (_req, res) => {
        res.once('close', disconnected.resolve);
        openStream(res);
        res.write(`data: ${event('first')}\n\n`);
      },
      async (client) => {
        for await (const chunk of client._streamResponseChunks(
          [new HumanMessage('Hello')],
          {}
        )) {
          expect(chunk.text).toBe('first');
          break;
        }
        await expectDisconnected(disconnected.promise);
      },
      25
    );
  }, 5000);

  it('preserves the status and message of an initial Google HTTP rejection', async () => {
    await withGateway(
      (_req, res) => {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(
          JSON.stringify({
            error: {
              message:
                'Input token count exceeds the maximum number of tokens allowed (100)',
            },
          })
        );
      },
      async (client) => {
        await expect(client.stream('Hello')).rejects.toMatchObject({
          status: 400,
          message: expect.stringContaining(
            'maximum number of tokens allowed (100)'
          ),
        });
      }
    );
  });

  it.each([
    ['data: {private gateway payload}\n\n', 'Invalid JSON in Gemini stream'],
    ['data: {"candidates":', 'Incomplete Gemini stream event'],
    [
      'event: error\ndata: private gateway payload\n\n',
      'Gemini gateway returned a stream error',
    ],
    [
      'data: {"error":{"code":503,"message":"private gateway payload"}}\n\n',
      'Gemini gateway returned a stream error',
    ],
  ])(
    'rejects malformed and explicit gateway errors after streaming text',
    async (frame, message) => {
      await withGateway(
        (_req, res) => {
          openStream(res);
          res.end(`data: ${event('first')}\n\n${frame}`);
        },
        async (client) => {
          const stream = await client.stream('Hello');
          const iterator = stream[Symbol.asyncIterator]();
          expect((await iterator.next()).value?.text).toBe('first');
          await expect(iterator.next()).rejects.toThrow(message);
        }
      );
    }
  );

  it('rejects invalid UTF-8 instead of changing generated text', async () => {
    await withGateway(
      (_req, res) => {
        openStream(res);
        const frame = Buffer.from(`data: ${event('utf8-marker')}\n\n`);
        frame[frame.indexOf('utf8-marker')] = 0xff;
        res.end(frame);
      },
      async (client) => {
        const consume = async () =>
          Array.fromAsync(await client.stream('Hello'));
        await expect(consume()).rejects.toThrow(/utf-8/i);
      }
    );
  });
});
