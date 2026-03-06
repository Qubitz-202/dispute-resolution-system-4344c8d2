import * as cdk from 'aws-cdk-lib';
import * as agentcore from '@aws-cdk/aws-bedrock-agentcore-alpha';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as nodejs from 'aws-cdk-lib/aws-lambda-nodejs';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import { Asset } from 'aws-cdk-lib/aws-s3-assets';
import * as path from 'path';
import { Construct } from 'constructs';

export interface AgentCoreStackProps extends cdk.StackProps {
  stage: string;
  agentName: string;
  modelId: string;
  enableMemory: boolean;
  memoryStrategy: string;
  enableGateway: boolean;
  userId?: string;
  projectId?: string;
}

export class AgentCoreStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: AgentCoreStackProps) {
    super(scope, id, props);

    const { stage, agentName, modelId, enableMemory, memoryStrategy, enableGateway, userId, projectId } = props;

    // ── IAM Role (full access for agent runtime) ─────────────────
    const agentRole = new iam.Role(this, 'AgentRole', {
      roleName: `${agentName}-agentcore-role`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
        new iam.ServicePrincipal('bedrock.amazonaws.com'),
      ),
    });

    agentRole.addToPolicy(new iam.PolicyStatement({
      actions: ['*'],
      resources: ['*'],
    }));

    agentRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('AWSXRayDaemonWriteAccess')
    );

    // ── Memory (optional) ────────────────────────────────────────
    let memory: agentcore.Memory | undefined;
    if (enableMemory) {
      let strategies: agentcore.ManagedMemoryStrategy[] = [];
      switch (memoryStrategy) {
        case 'SUMMARY':
          strategies = [agentcore.MemoryStrategy.usingBuiltInSummarization()]; break;
        case 'USER_PREFERENCE':
          strategies = [agentcore.MemoryStrategy.usingBuiltInUserPreference()]; break;
        case 'SEMANTIC': default:
          strategies = [agentcore.MemoryStrategy.usingBuiltInSemantic()];
      }

      memory = new agentcore.Memory(this, 'Memory', {
        memoryName: `${agentName}Memory`,
        description: `Memory for ${agentName}`,
        expirationDuration: cdk.Duration.days(90),
        memoryStrategies: strategies,
      });

      new cdk.CfnOutput(this, 'MemoryName', { value: memory.memoryName });
      new cdk.CfnOutput(this, 'MemoryArn', { value: memory.memoryArn });
    }

    // ── AgentCore Runtime ────────────────────────────────────────
    const codeAsset = new Asset(this, 'CodeAsset', {
      path: path.join(__dirname, '../../agent'),
    });

    const runtime = new agentcore.Runtime(this, 'Runtime', {
      runtimeName: `${agentName}Runtime`,
      description: `AgentCore Runtime for ${agentName}`,
      executionRole: agentRole,
      agentRuntimeArtifact: agentcore.AgentRuntimeArtifact.fromS3(
        { bucketName: codeAsset.s3BucketName, objectKey: codeAsset.s3ObjectKey },
        agentcore.AgentCoreRuntime.PYTHON_3_12,
        ['agent.py'],
      ),
      networkConfiguration: agentcore.RuntimeNetworkConfiguration.usingPublicNetwork(),
      environmentVariables: {
        STAGE: stage,
        BEDROCK_MODEL_ID: modelId,
        LOG_LEVEL: stage === 'prod' ? 'INFO' : 'DEBUG',
        AWS_REGION: this.region,
        AWS_DEFAULT_REGION: this.region,
        DYNAMODB_TABLE_NAME: 'AgentConfigs',
        CREDENTIALS_TABLE_NAME: 'ToolCredentials',
        ...(userId ? { USER_ID: userId } : {}),
        ...(projectId ? { PROJECT_ID: projectId } : {}),
        ...(enableMemory && memory ? { AGENTCORE_MEMORY_ID: memory.memoryId } : {}),
      },
    });

    // ── Logs ─────────────────────────────────────────────────────
    new logs.LogGroup(this, 'LogGroup', {
      logGroupName: `/aws/bedrock-agentcore/runtimes/${agentName}`,
      retention: logs.RetentionDays.TWO_WEEKS,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ── Streaming Lambda + REST API Gateway ──────────────────────
    const streamingLambda = new nodejs.NodejsFunction(this, 'StreamingProxy', {
      functionName: `${agentName}-streaming-proxy`,
      entry: path.join(__dirname, '../lambda/streaming-proxy/index.ts'),
      handler: 'handler',
      runtime: lambda.Runtime.NODEJS_22_X,
      timeout: cdk.Duration.minutes(15),
      memorySize: 256,
      environment: {
        AGENT_RUNTIME_ARN: runtime.agentRuntimeArn,
        AWS_REGION_NAME: this.region,
      },
      bundling: { externalModules: [] },
    });

    streamingLambda.addToRolePolicy(new iam.PolicyStatement({
      actions: ['bedrock-agentcore:InvokeAgentRuntime'],
      resources: [runtime.agentRuntimeArn, `${runtime.agentRuntimeArn}/*`],
    }));

    const api = new apigateway.RestApi(this, 'AgentRestApi', {
      restApiName: `${agentName}-api`,
      description: `API for ${agentName}`,
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: ['POST', 'OPTIONS'],
        allowHeaders: ['Content-Type', 'Authorization', 'X-Api-Key'],
      },
      deployOptions: { stageName: 'prod' },
    });

    const postMethod = api.root.addResource('invoke').addMethod(
      'POST',
      new apigateway.LambdaIntegration(streamingLambda, { proxy: true }),
    );

    // Enable response streaming
    const cfnMethod = postMethod.node.defaultChild as apigateway.CfnMethod;
    cfnMethod.addPropertyOverride('Integration.ResponseTransferMode', 'STREAM');
    cfnMethod.addPropertyOverride('Integration.TimeoutInMillis', 900000);
    cfnMethod.addPropertyOverride(
      'Integration.Uri',
      cdk.Fn.sub(
        'arn:aws:apigateway:${AWS::Region}:lambda:path/2021-11-15/functions/${FnArn}/response-streaming-invocations',
        { FnArn: streamingLambda.functionArn },
      ),
    );

    // ── Custom Domain: api.qubitz.ai/{agentName}/invoke ─────────
    const customDomain = apigateway.DomainName.fromDomainNameAttributes(this, 'ApiDomain', {
      domainName: 'api.qubitz.ai',
      domainNameAliasHostedZoneId: 'Z1U9ULNL0V5AJ3',
      domainNameAliasTarget: 'd-n547jsqln6.execute-api.eu-central-1.amazonaws.com',
    });

    new apigateway.BasePathMapping(this, 'BasePathMapping', {
      domainName: customDomain,
      restApi: api,
      basePath: agentName,
    });

    // ── Outputs ──────────────────────────────────────────────────
    new cdk.CfnOutput(this, 'RuntimeArn', { value: runtime.agentRuntimeArn, exportName: `${id}-RuntimeArn` });
    new cdk.CfnOutput(this, 'RuntimeName', { value: runtime.agentRuntimeName });
    new cdk.CfnOutput(this, 'RuntimeId', { value: runtime.agentRuntimeId });
    new cdk.CfnOutput(this, 'AgentRoleArn', { value: agentRole.roleArn });
    new cdk.CfnOutput(this, 'ApiEndpoint', { value: `https://api.qubitz.ai/${agentName}/invoke`, exportName: `${id}-ApiEndpoint` });
    new cdk.CfnOutput(this, 'ApiEndpointRaw', { value: `${api.url}invoke` });
    new cdk.CfnOutput(this, 'ApiId', { value: api.restApiId });
  }
}
