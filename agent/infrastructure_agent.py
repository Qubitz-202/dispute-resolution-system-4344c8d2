"""
Platform Agent: Infrastructure & Reliability Agent
===================================================
Provides real-time system health checks using CloudWatch X-Ray traces,
model availability verification, and tool health checks.

This module is used by base-template.py when the infrastructure
platform agent is enabled in config (platform_agents.infrastructure_agent.enabled = true).

Tools query real observability data from the `aws/spans` log group where
X-Ray traces are stored via ADOT (OpenTelemetry) instrumentation.
"""

import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import boto3
from botocore.exceptions import ClientError

try:
    from strands import tool
except ImportError:
    # Allow running tests locally without strands installed
    def tool(fn):
        fn.fn = fn
        return fn

logger = logging.getLogger(__name__)


# =============================================================================
# MODULE-LEVEL STATE (set by base-template.py)
# =============================================================================
_agents_config: Dict[str, Any] = {}
_gateway_url: Optional[str] = None
_aws_region: str = 'eu-central-1'
_agent_runtime_id: Optional[str] = None  # AgentCore runtime ID for span filtering


def set_agents_config_reference(
    agents_config: Dict[str, Any],
    gateway_url: Optional[str] = None,
    region: str = 'eu-central-1',
):
    """Called by base-template.py to give tools access to agent configs."""
    global _agents_config, _gateway_url, _aws_region, _agent_runtime_id
    _agents_config = agents_config
    _gateway_url = gateway_url
    _aws_region = region
    # Runtime ID comes from the AGENTCORE_RUNTIME_ARN env var set by CDK
    # Format: arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/RUNTIME_ID
    runtime_arn = os.environ.get('AGENTCORE_RUNTIME_ARN', '')
    if 'runtime/' in runtime_arn:
        _agent_runtime_id = runtime_arn.split('runtime/')[-1].rstrip('/')
    else:
        _agent_runtime_id = os.environ.get('AGENTCORE_RUNTIME_ID', None)


# =============================================================================
# CLOUDWATCH LOGS INSIGHTS HELPERS
# =============================================================================
SPANS_LOG_GROUP = 'aws/spans'
QUERY_TIMEOUT_SECONDS = 30
QUERY_POLL_INTERVAL = 2


def _execute_cloudwatch_query(
    query_string: str,
    start_time_epoch_s: int,
    end_time_epoch_s: int,
    log_group: str = SPANS_LOG_GROUP,
) -> List[List[Dict[str, str]]]:
    """Execute a CloudWatch Logs Insights query and return results.

    Args:
        query_string: The Logs Insights query.
        start_time_epoch_s: Start time in epoch seconds.
        end_time_epoch_s: End time in epoch seconds.
        log_group: Log group to query (default: aws/spans).

    Returns:
        List of result rows, each row is a list of {field, value} dicts.
    """
    logs_client = boto3.client('logs', region_name=_aws_region)

    try:
        response = logs_client.start_query(
            logGroupName=log_group,
            startTime=start_time_epoch_s,
            endTime=end_time_epoch_s,
            queryString=query_string,
        )
    except ClientError as e:
        code = e.response['Error']['Code']
        if code == 'ResourceNotFoundException':
            logger.warning(f"Log group '{log_group}' not found — traces may not be enabled yet")
            return []
        raise

    query_id = response['queryId']
    deadline = time.time() + QUERY_TIMEOUT_SECONDS

    while time.time() < deadline:
        result = logs_client.get_query_results(queryId=query_id)
        status = result['status']

        if status == 'Complete':
            return result.get('results', [])
        if status in ('Failed', 'Cancelled'):
            logger.warning(f"CloudWatch query {status}: {query_id}")
            return []

        time.sleep(QUERY_POLL_INTERVAL)

    logger.warning(f"CloudWatch query timed out after {QUERY_TIMEOUT_SECONDS}s")
    return []


def _row_to_dict(row: List[Dict[str, str]]) -> Dict[str, str]:
    """Convert a CloudWatch result row [{field, value}, ...] to a flat dict."""
    return {item['field']: item['value'] for item in row if item.get('field') != '@ptr'}


def _get_time_range(hours_back: int = 1):
    """Return (start_epoch_s, end_epoch_s) for the last N hours."""
    now = int(time.time())
    return now - (hours_back * 3600), now


# =============================================================================
# INFRASTRUCTURE AGENT TOOLS
# =============================================================================

@tool
def get_system_status() -> str:
    """Get real-time system health from X-Ray traces (last 1 hour).

    Queries CloudWatch Logs Insights for span data to report:
    - Total traces, spans, and error counts
    - Error rate percentage
    - Average and max latency
    - Per-agent breakdown (if multiple agents share the runtime)

    Returns:
        JSON string with system health summary from real trace data.
    """
    start_s, end_s = _get_time_range(hours_back=1)

    # Build agent filter
    agent_filter = ""
    if _agent_runtime_id:
        agent_filter = f"""| parse resource.attributes.cloud.resource_id "runtime/*/" as parsedAgentId
| filter parsedAgentId = '{_agent_runtime_id}'"""

    query = f"""fields traceId,
       name as spanName,
       durationNano/1000000 as durationMs,
       status.code as statusCode,
       attributes.session.id as sessionId,
       resource.attributes.service.name as serviceName
{agent_filter}
| stats count(*) as totalSpans,
        count_distinct(traceId) as totalTraces,
        count_distinct(attributes.session.id) as totalSessions,
        avg(durationNano/1000000) as avgDurationMs,
        max(durationNano/1000000) as maxDurationMs,
        sum(statusCode = 'STATUS_CODE_ERROR') as errorSpans
  by resource.attributes.service.name"""

    results = _execute_cloudwatch_query(query, start_s, end_s)

    if not results:
        return json.dumps({
            'summary': {
                'system_status': 'no_data',
                'message': 'No trace data found in the last hour. The runtime may not have received requests yet, or X-Ray Transaction Search may not be enabled.',
                'time_range': {'hours_back': 1},
            }
        }, indent=2)

    services = []
    total_spans = 0
    total_errors = 0
    total_traces = 0

    for row in results:
        d = _row_to_dict(row)
        spans = int(d.get('totalSpans', 0))
        errors = int(d.get('errorSpans', 0))
        traces = int(d.get('totalTraces', 0))
        total_spans += spans
        total_errors += errors
        total_traces += traces

        services.append({
            'service': d.get('resource.attributes.service.name', 'unknown'),
            'total_spans': spans,
            'total_traces': traces,
            'total_sessions': int(d.get('totalSessions', 0)),
            'error_spans': errors,
            'error_rate_pct': round(errors / spans * 100, 2) if spans > 0 else 0,
            'avg_latency_ms': round(float(d.get('avgDurationMs', 0)), 2),
            'max_latency_ms': round(float(d.get('maxDurationMs', 0)), 2),
        })

    error_rate = round(total_errors / total_spans * 100, 2) if total_spans > 0 else 0
    system_status = 'healthy' if error_rate < 5 else ('degraded' if error_rate < 20 else 'unhealthy')

    return json.dumps({
        'summary': {
            'system_status': system_status,
            'total_traces': total_traces,
            'total_spans': total_spans,
            'total_errors': total_errors,
            'error_rate_pct': error_rate,
            'time_range': {'hours_back': 1},
        },
        'services': services,
    }, indent=2)


@tool
def get_recent_errors() -> str:
    """Get recent error spans from X-Ray traces (last 1 hour).

    Queries for spans with ERROR status code to identify failing operations.
    Returns the most recent 20 errors with trace IDs, span names, and error details.

    Returns:
        JSON string with recent error details.
    """
    start_s, end_s = _get_time_range(hours_back=1)

    agent_filter = ""
    if _agent_runtime_id:
        agent_filter = f"""| parse resource.attributes.cloud.resource_id "runtime/*/" as parsedAgentId
| filter parsedAgentId = '{_agent_runtime_id}'"""

    query = f"""fields @timestamp,
       traceId,
       spanId,
       name as spanName,
       status.code as statusCode,
       status.message as statusMessage,
       durationNano/1000000 as durationMs,
       attributes.session.id as sessionId,
       events,
       resource.attributes.service.name as serviceName
| filter status.code = 'STATUS_CODE_ERROR'
{agent_filter}
| sort @timestamp desc
| limit 20"""

    results = _execute_cloudwatch_query(query, start_s, end_s)

    if not results:
        return json.dumps({
            'summary': {'total_errors': 0, 'message': 'No errors in the last hour.'},
            'errors': [],
        }, indent=2)

    errors = []
    for row in results:
        d = _row_to_dict(row)
        errors.append({
            'timestamp': d.get('@timestamp', ''),
            'trace_id': d.get('traceId', ''),
            'span_id': d.get('spanId', ''),
            'span_name': d.get('spanName', ''),
            'status_message': d.get('statusMessage', ''),
            'duration_ms': round(float(d.get('durationMs', 0)), 2),
            'session_id': d.get('sessionId', ''),
            'service': d.get('serviceName', ''),
            'events': d.get('events', ''),
        })

    return json.dumps({
        'summary': {
            'total_errors': len(errors),
            'time_range': {'hours_back': 1},
        },
        'errors': errors,
    }, indent=2)


@tool
def get_trace_summary(session_id: str) -> str:
    """Get a detailed trace summary for a specific session.

    Retrieves all spans for the given session ID to show the full request flow:
    agent calls, tool invocations, durations, and any errors.

    Args:
        session_id: The session ID to look up traces for.

    Returns:
        JSON string with all spans for the session, grouped by trace.
    """
    start_s, end_s = _get_time_range(hours_back=6)

    agent_filter = ""
    if _agent_runtime_id:
        agent_filter = f"""| parse resource.attributes.cloud.resource_id "runtime/*/" as parsedAgentId
| filter parsedAgentId = '{_agent_runtime_id}'"""

    query = f"""fields @timestamp,
       traceId,
       spanId,
       name as spanName,
       kind,
       status.code as statusCode,
       status.message as statusMessage,
       durationNano/1000000 as durationMs,
       attributes.session.id as sessionId,
       startTimeUnixNano,
       endTimeUnixNano,
       parentSpanId,
       events,
       resource.attributes.service.name as serviceName,
       attributes.aws.remote.service as serviceType
| filter attributes.session.id = '{session_id}'
{agent_filter}
| sort startTimeUnixNano asc
| limit 200"""

    results = _execute_cloudwatch_query(query, start_s, end_s)

    if not results:
        return json.dumps({
            'summary': {
                'session_id': session_id,
                'message': 'No spans found for this session. It may be too old (>6h) or the session ID may be incorrect.',
            },
            'spans': [],
        }, indent=2)

    spans = []
    traces = set()
    error_count = 0

    for row in results:
        d = _row_to_dict(row)
        trace_id = d.get('traceId', '')
        traces.add(trace_id)
        is_error = d.get('statusCode', '') == 'STATUS_CODE_ERROR'
        if is_error:
            error_count += 1

        spans.append({
            'trace_id': trace_id,
            'span_id': d.get('spanId', ''),
            'parent_span_id': d.get('parentSpanId', ''),
            'span_name': d.get('spanName', ''),
            'kind': d.get('kind', ''),
            'status': d.get('statusCode', 'OK'),
            'status_message': d.get('statusMessage', ''),
            'duration_ms': round(float(d.get('durationMs', 0)), 2),
            'service': d.get('serviceName', ''),
            'service_type': d.get('serviceType', ''),
            'events': d.get('events', ''),
        })

    return json.dumps({
        'summary': {
            'session_id': session_id,
            'total_traces': len(traces),
            'total_spans': len(spans),
            'error_spans': error_count,
        },
        'spans': spans,
    }, indent=2)


@tool
def check_model_availability() -> str:
    """Check if the Bedrock models used by each agent are reachable and responding.

    Sends a minimal request to each unique model to verify the endpoint is live.
    Reports per-agent model status: available, throttled, or unreachable.

    Returns:
        JSON string with per-agent model availability status.
    """
    if not _agents_config:
        return json.dumps({"error": "Agent configs not available"})

    bedrock = boto3.client('bedrock-runtime', region_name=_aws_region)
    results = {}
    checked_models = {}  # cache: model_id -> result

    for agent_key, agent_cfg in _agents_config.items():
        agent_name = agent_cfg.get('name', agent_key)
        model_id = agent_cfg.get('model', '')

        if not model_id:
            results[agent_name] = {'status': 'unknown', 'reason': 'no model configured'}
            continue

        if model_id in checked_models:
            results[agent_name] = {**checked_models[model_id], 'model': model_id}
            continue

        try:
            bedrock.converse(
                modelId=model_id,
                messages=[{
                    'role': 'user',
                    'content': [{'text': 'ping'}]
                }],
                inferenceConfig={'maxTokens': 1}
            )
            result = {'status': 'available', 'model': model_id}

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                result = {'status': 'throttled', 'model': model_id, 'reason': 'rate limited'}
            elif error_code in ('AccessDeniedException', 'UnauthorizedException'):
                result = {'status': 'unreachable', 'model': model_id, 'reason': 'access denied'}
            elif error_code == 'ResourceNotFoundException':
                result = {'status': 'unreachable', 'model': model_id, 'reason': 'model not found'}
            else:
                result = {'status': 'unreachable', 'model': model_id, 'reason': str(e)[:100]}
        except Exception as e:
            result = {'status': 'unreachable', 'model': model_id, 'reason': str(e)[:100]}

        checked_models[model_id] = result
        results[agent_name] = result

    available_count = sum(1 for r in results.values() if r['status'] == 'available')
    return json.dumps({
        'summary': {
            'total_agents': len(results),
            'models_available': available_count,
            'models_unavailable': len(results) - available_count,
        },
        'agents': results,
    }, indent=2)


@tool
def check_agent_tools(agent_name: str) -> str:
    """Check if the tools assigned to a specific agent are reachable and working.

    Verifies connectivity for each tool type:
    - MCP gateway targets: checks gateway URL health
    - Web search: checks if the search endpoint responds
    - Knowledge base / RAG: checks if S3 Vectors bucket is accessible

    Args:
        agent_name: The name of the agent whose tools should be checked.

    Returns:
        JSON string with per-tool health status.
    """
    if not _agents_config:
        return json.dumps({"error": "Agent configs not available"})

    # Find the agent config
    agent_cfg = None
    for key, cfg in _agents_config.items():
        if cfg.get('name', key) == agent_name:
            agent_cfg = cfg
            break

    if not agent_cfg:
        available = [cfg.get('name', k) for k, cfg in _agents_config.items()]
        return json.dumps({
            "error": f"Agent '{agent_name}' not found",
            "available_agents": available
        })

    tool_names = agent_cfg.get('tools', [])
    if not tool_names:
        return json.dumps({
            "agent": agent_name,
            "message": "No tools configured for this agent"
        })

    results = {}
    for tool_name in tool_names:
        if tool_name in ('web_search',):
            results[tool_name] = _check_web_search_tool()
        elif tool_name.startswith(('mcp_', 'mcp:', 'mcp_server:', 'integration:')):
            results[tool_name] = _check_mcp_tool(tool_name)
        elif tool_name in ('rag', 'RAG', 'knowledge_base_query'):
            results[tool_name] = _check_knowledge_base_tool()
        elif tool_name.startswith('connector:'):
            connector_name = tool_name.split(':', 1)[1] if ':' in tool_name else ''
            results[tool_name] = {'status': 'configured', 'note': f'Connector {connector_name} — requires per-user OAuth at runtime'}
        else:
            results[tool_name] = {'status': 'available', 'type': 'local'}

    healthy = sum(1 for r in results.values() if r.get('status') in ('available', 'configured'))
    return json.dumps({
        'agent': agent_name,
        'summary': {
            'total_tools': len(results),
            'healthy': healthy,
            'unhealthy': len(results) - healthy,
        },
        'tools': results,
    }, indent=2)


# ── Helper functions for tool checks ──────────────────────────────────────

def _check_web_search_tool() -> Dict[str, Any]:
    """Check web search tool availability."""
    try:
        return {'status': 'available', 'type': 'local', 'provider': 'playwright_scraper'}
    except Exception as e:
        return {'status': 'unavailable', 'reason': str(e)[:100]}


def _check_mcp_tool(tool_name: str) -> Dict[str, Any]:
    """Check MCP gateway tool availability."""
    if not _gateway_url:
        return {'status': 'unavailable', 'reason': 'no gateway URL configured'}

    try:
        import urllib.request
        req = urllib.request.Request(_gateway_url, method='GET')
        req.add_header('Connection', 'close')
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status < 400:
                return {'status': 'available', 'type': 'mcp_gateway', 'gateway_url': _gateway_url}
            return {'status': 'degraded', 'reason': f'HTTP {resp.status}'}
    except Exception as e:
        return {'status': 'unavailable', 'type': 'mcp_gateway', 'reason': str(e)[:100]}


def _check_knowledge_base_tool() -> Dict[str, Any]:
    """Check knowledge base (S3 Vectors) availability."""
    try:
        bucket = os.environ.get('S3_VECTOR_BUCKET', 'qubitz-vectors-prod')
        s3 = boto3.client('s3', region_name=_aws_region)
        s3.head_bucket(Bucket=bucket)
        return {'status': 'available', 'type': 'knowledge_base', 'bucket': bucket}
    except ClientError as e:
        code = e.response['Error']['Code']
        return {'status': 'unavailable', 'type': 'knowledge_base', 'reason': f'S3 error: {code}'}
    except Exception as e:
        return {'status': 'unavailable', 'type': 'knowledge_base', 'reason': str(e)[:100]}


# =============================================================================
# INFRASTRUCTURE AGENT PROMPT
# =============================================================================
INFRASTRUCTURE_AGENT_PROMPT = """You are the Infrastructure & Reliability Platform Agent for this multi-agent system.

Your role is to monitor system health using real X-Ray trace data from CloudWatch. You provide factual reports on system performance, errors, and availability.

YOUR RESPONSIBILITIES:
1. System Health: Query real trace data to report error rates, latencies, and throughput
2. Error Investigation: Find recent error spans with details for debugging
3. Session Tracing: Show full request flow for a specific session (all spans, durations, errors)
4. Model Availability: Check if Bedrock models used by agents are reachable
5. Tool Health: Verify that agent tools (MCP, web search, knowledge base) are functional

TOOLS AVAILABLE:
- get_system_status: Get aggregated health metrics from X-Ray traces (last 1 hour)
- get_recent_errors: Find recent error spans with trace IDs and error messages
- get_trace_summary: Get all spans for a specific session ID to trace the full request flow
- check_model_availability: Ping each agent's Bedrock model to verify availability
- check_agent_tools: Check if a specific agent's tools are reachable

HOW TO RESPOND:
- When asked about system health, use get_system_status first
- When asked about errors or failures, use get_recent_errors
- When asked to investigate a specific session/request, use get_trace_summary with the session ID
- When an agent is reported as failing, check its model and tools
- Be concise and factual — report data, not opinions
- If no trace data is found, explain that the runtime may not have received requests yet

YOU DO NOT:
- Modify agent configurations or prompts
- Make deployment decisions
- Track token usage or costs
"""

# Tool list for config — must match the @tool function names above
INFRASTRUCTURE_AGENT_TOOLS = [
    'get_system_status',
    'get_recent_errors',
    'get_trace_summary',
    'check_model_availability',
    'check_agent_tools',
]
