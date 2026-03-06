
declare const awslambda: {
  streamifyResponse: (handler: Function) => Function;
  HttpResponseStream: {
    from: (responseStream: any, metadata: any) => any;
  };
};

import {
  BedrockAgentCoreClient,
  InvokeAgentRuntimeCommand,
} from '@aws-sdk/client-bedrock-agentcore';
import { pipeline } from 'stream/promises';
import { Readable, Writable } from 'stream';
import { randomUUID } from 'crypto';

const arn = process.env.AGENT_RUNTIME_ARN!;
const region = process.env.AWS_REGION_NAME || 'eu-central-1';
const client = new BedrockAgentCoreClient({ region });

export const handler = awslambda.streamifyResponse(
  async (event: any, responseStream: any, _context: any) => {
    try {
      const body = typeof event.body === 'string' ? JSON.parse(event.body) : event.body || {};
      const prompt: string = body.prompt || '';
      const sid: string = body.session_id || randomUUID();
      const aid: string = body.actor_id || process.env.USER_ID || 'anonymous';
      const stream: boolean = body.stream !== undefined ? body.stream : true;

      if (!prompt) {
        responseStream = awslambda.HttpResponseStream.from(responseStream, {
          statusCode: 400,
          headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        });
        responseStream.write(JSON.stringify({ error: 'prompt is required' }));
        responseStream.end();
        return;
      }

      const r = await client.send(new InvokeAgentRuntimeCommand({
        agentRuntimeArn: arn,
        runtimeSessionId: sid,
        payload: new TextEncoder().encode(JSON.stringify({ prompt, session_id: sid, actor_id: aid, stream })),
        contentType: 'application/json',
        qualifier: 'DEFAULT',
      }));

      responseStream = awslambda.HttpResponseStream.from(responseStream, {
        statusCode: 200,
        headers: {
          'Content-Type': stream ? 'text/event-stream' : 'application/json',
          'Cache-Control': 'no-cache',
          'Access-Control-Allow-Origin': '*',
        },
      });

      if (r.response) {
        await pipeline(r.response as unknown as Readable, responseStream as unknown as Writable);
      } else {
        responseStream.write(JSON.stringify({ error: 'No response from runtime' }));
        responseStream.end();
      }
    } catch (e: any) {
      try {
        responseStream = awslambda.HttpResponseStream.from(responseStream, {
          statusCode: 500,
          headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        });
        responseStream.write(JSON.stringify({ error: e.message }));
        responseStream.end();
      } catch {}
    }
  },
);
