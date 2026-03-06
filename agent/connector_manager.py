"""Runtime connector manager — per-user OAuth tokens from AgentCore Identity vault.

Used by the deployed agent runtime (base-template.py) to:
  1. Get a user's OAuth token from Identity vault
  2. Create MCP clients with that user's Bearer token
  3. Return MCP tools for the agent to use

Each end-user authenticates their own Slack/Google/etc account.
Tokens are stored per-user in Identity vault — this manager retrieves them at request time.
"""

import os
import logging
import boto3
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

WORKLOAD_IDENTITY_NAME = os.environ.get('WORKLOAD_IDENTITY_NAME', '')
AWS_REGION = os.environ.get('AWS_REGION', 'eu-central-1')


class ConnectorManager:
    """Gets per-user OAuth tokens from AgentCore Identity and creates MCP clients."""

    def __init__(self):
        self._data_client = None
        self._workload_tokens = {}

    @property
    def data_client(self):
        if self._data_client is None:
            self._data_client = boto3.client('bedrock-agentcore', region_name=AWS_REGION)
        return self._data_client

    def _get_workload_token(self, user_id: str) -> str:
        """Get (or cache) a workload access token for a user."""
        if user_id in self._workload_tokens:
            return self._workload_tokens[user_id]

        response = self.data_client.get_workload_access_token_for_user_id(
            workloadName=WORKLOAD_IDENTITY_NAME,
            userId=user_id,
        )
        token = response["workloadAccessToken"]
        self._workload_tokens[user_id] = token
        logger.info(f"Got workload token for user={user_id}")
        return token

    def get_user_token(self, user_id: str, provider_name: str, scopes: list) -> Optional[str]:
        """Get a user's OAuth token from Identity vault.

        Returns the access token string, or None if not authorized.
        AgentCore handles token refresh automatically.
        """
        try:
            workload_token = self._get_workload_token(user_id)
            response = self.data_client.get_resource_oauth2_token(
                workloadIdentityToken=workload_token,
                resourceCredentialProviderName=provider_name,
                scopes=scopes,
                oauth2Flow="USER_FEDERATION",
                forceAuthentication=False,
            )
            return response.get("accessToken")
        except Exception as e:
            logger.error(f"Failed to get token for {provider_name}/{user_id}: {e}")
            return None

    def get_connector_tools(
        self, connector_configs: List[dict], user_id: str
    ) -> Tuple[List, List]:
        """Get MCP tools for all connectors for a user.

        Args:
            connector_configs: List of connector target dicts from CONNECTOR_TARGETS env var.
                Each has: identity_provider, mcp_endpoint, scopes
            user_id: The end-user's ID (actor_id from the request)

        Returns:
            (tools, mcp_clients) — caller must stop mcp_clients after use.
        """
        from strands.tools.mcp import MCPClient
        from mcp.client.streamable_http import streamablehttp_client

        all_tools = []
        mcp_clients = []
        tools_by_connector = {}  # name → [tool_name, ...]

        for config in connector_configs:
            provider = config.get('identity_provider', '')
            endpoint = config.get('mcp_endpoint', '')
            scopes = config.get('scopes', [])
            name = config.get('name', provider)

            if not provider or not endpoint:
                logger.warning(f"Connector '{name}' missing provider or endpoint, skipping")
                continue

            token = self.get_user_token(user_id, provider, scopes)
            if not token:
                logger.warning(f"No token for connector '{name}', user {user_id} — not authorized")
                continue

            try:
                mcp_client = MCPClient(
                    lambda ep=endpoint, tk=token: streamablehttp_client(
                        url=ep, headers={"Authorization": f"Bearer {tk}"}
                    )
                )
                mcp_client.start()
                tools = mcp_client.list_tools_sync()
                all_tools.extend(tools)
                mcp_clients.append(mcp_client)
                tools_by_connector[name] = [
                    getattr(t, 'tool_name', getattr(t, '__name__', str(t))) for t in tools
                ]
                logger.info(f"Loaded {len(tools)} tools from connector '{name}'")
            except Exception as e:
                logger.error(f"Failed to connect to connector '{name}' at {endpoint}: {e}")

        return all_tools, mcp_clients, tools_by_connector


# Singleton
_connector_manager: Optional[ConnectorManager] = None


def get_connector_manager() -> Optional[ConnectorManager]:
    """Get or create the ConnectorManager singleton."""
    global _connector_manager
    if _connector_manager is None and WORKLOAD_IDENTITY_NAME:
        _connector_manager = ConnectorManager()
    return _connector_manager
