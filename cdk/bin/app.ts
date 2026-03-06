#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { AgentCoreStack } from '../lib/agentcore-stack';

const app = new cdk.App();

const stage = process.env.STAGE || 'prod';
const account = process.env.CDK_DEFAULT_ACCOUNT || '599138915470';
const region = process.env.CDK_DEFAULT_REGION || 'eu-central-1';

new AgentCoreStack(app, `dispute-resolution-system-${stage}`, {
  stage,
  agentName: 'dispute_resolution_system_4344c8d2',
  modelId: 'eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
  enableMemory: true,
  memoryStrategy: 'SEMANTIC',
  enableGateway: false,
  userId: '4344c8d2-7091-7071-4b83-c8626f6a3e18',
  projectId: '41221f51-8f1d-4e99-bac7-088d248b9ebf',
  env: { account, region },
});

app.synth();
