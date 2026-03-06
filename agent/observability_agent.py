"""
Platform Agent: Observability & Telemetry Agent
=================================================
Retrieves runtime telemetry data by calling an external observability agent
deployed on AWS Bedrock AgentCore. Provides aggregated metrics, session-level
details, and per-user usage breakdowns.

This module is used by base-template.py when the observability platform
agent is enabled in config (platform_agents.observability_agent.enabled = true).

The external observability runtime (observability-B0cewXH6Su) is invoked via
boto3 bedrock-agentcore invoke_agent_runtime API. It accepts a JSON payload
with agent_arn (required), get_session (optional), and user_id (optional).
Response is an SSE stream — status events followed by a final JSON data line.
"""

import json
import logging
from typing import Dict, Any, Optional

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
_runtime_arn: Optional[str] = None  # This runtime's own ARN (passed to observability agent)

# External observability agent runtime ARN
_OBSERVABILITY_RUNTIME_ARN = 'arn:aws:bedrock-agentcore:eu-central-1:599138915470:runtime/observability-B0cewXH6Su'


def set_agents_config_reference(
    agents_config: Dict[str, Any],
    gateway_url: Optional[str] = None,
    region: str = 'eu-central-1',
    deployment_outputs: Optional[Dict[str, Any]] = None,
):
    """Called by base-template.py to give tools access to agent configs.

    Args:
        agents_config: The agents_config dict from DynamoDB.
        gateway_url: The AgentCore gateway URL.
        region: AWS region.
        deployment_outputs: CDK deployment outputs including runtime_arn.
    """
    global _agents_config, _gateway_url, _aws_region, _runtime_arn
    _agents_config = agents_config
    _gateway_url = gateway_url
    _aws_region = region

    # Get this runtime's own ARN from deployment_outputs (stored in DynamoDB after deploy)
    if deployment_outputs and isinstance(deployment_outputs, dict):
        _runtime_arn = deployment_outputs.get('runtime_arn', '')
        if _runtime_arn:
            logger.info(f"[OBSERVABILITY] Runtime ARN set: {_runtime_arn[:60]}...")
        else:
            logger.warning("[OBSERVABILITY] deployment_outputs found but runtime_arn is empty")
    else:
        logger.warning("[OBSERVABILITY] No deployment_outputs — runtime_arn not available (pre-deploy?)")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _invoke_observability_runtime(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke the external observability AgentCore runtime and parse the SSE response.

    The observability runtime accepts:
        agent_arn (required): ARN of the runtime to get telemetry for
        get_session (optional): session ID for single-session mode
        user_id (optional): user ID for per-user isolation

    Response is an SSE stream with status lines followed by a final JSON data line.

    Args:
        payload: Dict with agent_arn and optional get_session/user_id.

    Returns:
        Parsed dict from the final JSON data line, or error dict.
    """
    try:
        client = boto3.client('bedrock-agentcore', region_name=_aws_region)

        response = client.invoke_agent_runtime(
            agentRuntimeArn=_OBSERVABILITY_RUNTIME_ARN,
            payload=json.dumps(payload).encode('utf-8'),
            contentType='application/json',
            accept='application/json',
        )

        # Read the StreamingBody
        body = response.get('response')
        if not body:
            return {'error': 'empty_response', 'message': 'No response body from observability runtime'}

        raw = body.read()
        text = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)

        if not text.strip():
            return {'error': 'empty_response', 'message': 'Empty response from observability runtime'}

        # Parse SSE stream. All data lines are: data: "<json-encoded-string>"
        # Status events: data: "data: {\"status\": \"started\", ...}\n\n"
        # Final result:  data: "{\"delta\": true, ...}"
        # Both are JSON-encoded strings. Status lines decode to "data: {...}"
        # while the final result decodes to raw JSON that needs a second parse.
        final_data = None
        for line in text.split('\n'):
            line = line.strip()
            if not line.startswith('data: '):
                continue
            data_content = line[6:]  # strip "data: " prefix
            try:
                decoded = json.loads(data_content)
            except json.JSONDecodeError:
                continue

            # If decoded is already a dict, use it directly
            if isinstance(decoded, dict):
                final_data = decoded
                continue

            # If decoded is a string, it's either a status event or the final JSON
            if isinstance(decoded, str):
                # Skip status event lines (they start with "data: ")
                if decoded.strip().startswith('data:'):
                    continue
                # Try to parse the string as JSON (double-encoded final result)
                try:
                    inner = json.loads(decoded)
                    if isinstance(inner, dict):
                        final_data = inner
                except json.JSONDecodeError:
                    continue

        if final_data:
            return final_data

        return {'error': 'parse_error', 'message': 'Could not parse final data from SSE stream', 'raw_length': len(text)}

    except ClientError as e:
        code = e.response['Error']['Code']
        msg = e.response['Error'].get('Message', str(e))
        logger.error(f"[OBSERVABILITY] invoke_agent_runtime failed: {code} — {msg}")
        return {'error': code, 'message': msg[:300]}
    except Exception as e:
        logger.error(f"[OBSERVABILITY] Unexpected error: {str(e)[:200]}")
        return {'error': 'unexpected', 'message': str(e)[:300]}


# =============================================================================
# OBSERVABILITY AGENT TOOLS
# =============================================================================

@tool
def get_runtime_metrics() -> str:
    """Get aggregated runtime telemetry metrics from the observability agent.

    Retrieves overall system metrics including invocation counts, latencies,
    error rates, token usage, and resource consumption. Provides a high-level
    overview of the runtime's operational health and performance.

    Returns:
        JSON string with aggregated runtime metrics including sessions,
        token usage by model, error rates, and per-span latency breakdowns.
    """
    if not _runtime_arn:
        return json.dumps({
            'error': 'not_deployed',
            'message': 'Runtime ARN not available. Deploy the system first to enable observability.',
        }, indent=2)

    result = _invoke_observability_runtime({'agent_arn': _runtime_arn})

    if 'error' in result:
        return json.dumps(result, indent=2)

    return json.dumps({
        'source': 'observability_agent',
        'runtime_arn': _runtime_arn,
        'data': result,
    }, indent=2)


@tool
def get_session_details(session_id: str) -> str:
    """Get detailed telemetry for a specific session from the observability agent.

    Retrieves per-session metrics including invocation count, token usage,
    latencies per span, error counts, and model usage for a single session.

    Args:
        session_id: The session ID to look up telemetry for.

    Returns:
        JSON string with session-level telemetry details.
    """
    if not session_id or not session_id.strip():
        return json.dumps({
            'error': 'invalid_input',
            'message': 'session_id is required.',
        }, indent=2)

    if not _runtime_arn:
        return json.dumps({
            'error': 'not_deployed',
            'message': 'Runtime ARN not available. Deploy the system first to enable observability.',
        }, indent=2)

    result = _invoke_observability_runtime({
        'agent_arn': _runtime_arn,
        'get_session': session_id,
    })

    if 'error' in result:
        return json.dumps(result, indent=2)

    return json.dumps({
        'source': 'observability_agent',
        'runtime_arn': _runtime_arn,
        'session_id': session_id,
        'data': result,
    }, indent=2)


@tool
def get_user_metrics(user_id: str) -> str:
    """Get usage metrics filtered by user from the observability agent.

    Retrieves per-user telemetry including session count, total invocations,
    token usage, error rates, and usage patterns for a specific user.

    Args:
        user_id: The user ID to filter metrics for.

    Returns:
        JSON string with user-level usage metrics.
    """
    if not user_id or not user_id.strip():
        return json.dumps({
            'error': 'invalid_input',
            'message': 'user_id is required.',
        }, indent=2)

    if not _runtime_arn:
        return json.dumps({
            'error': 'not_deployed',
            'message': 'Runtime ARN not available. Deploy the system first to enable observability.',
        }, indent=2)

    result = _invoke_observability_runtime({
        'agent_arn': _runtime_arn,
        'user_id': user_id,
    })

    if 'error' in result:
        return json.dumps(result, indent=2)

    return json.dumps({
        'source': 'observability_agent',
        'runtime_arn': _runtime_arn,
        'user_id': user_id,
        'data': result,
    }, indent=2)


# =============================================================================
# OBSERVABILITY AGENT PROMPT
# =============================================================================
OBSERVABILITY_AGENT_PROMPT = """You are the Observability & Telemetry Platform Agent for this multi-agent system.

Your role is to retrieve and present runtime telemetry data by querying an external observability agent. You provide factual reports on system performance, session traces, and user usage patterns.

YOUR RESPONSIBILITIES:
1. Runtime Metrics: Get aggregated metrics — invocation counts, latencies, error rates, token usage
2. Session Details: Look up detailed telemetry for a specific session including full trace and tool calls
3. User Metrics: Get per-user usage breakdowns — session counts, invocations, token usage, error rates

TOOLS AVAILABLE:
- get_runtime_metrics: Get overall aggregated telemetry metrics for the runtime
- get_session_details: Get detailed telemetry for a specific session ID
- get_user_metrics: Get usage metrics filtered by a specific user ID

HOW TO RESPOND:
- When asked about overall system performance or metrics, use get_runtime_metrics
- When asked about a specific session or request trace, use get_session_details with the session ID
- When asked about a specific user's usage, use get_user_metrics with the user ID
- Be concise and factual — report data, not opinions
- If the system is not yet deployed, explain that deployment is required first

YOU DO NOT:
- Modify agent configurations or telemetry settings
- Make deployment decisions
- Track costs directly (defer to the cost agent for cost analysis)
"""

# Tool list for config — must match the @tool function names above
OBSERVABILITY_AGENT_TOOLS = [
    'get_runtime_metrics',
    'get_session_details',
    'get_user_metrics',
]
