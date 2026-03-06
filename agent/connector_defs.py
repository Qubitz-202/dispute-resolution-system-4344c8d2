"""Pre-built connector definitions for user-level OAuth via AgentCore Identity.

Identity resources (workload identity + credential providers) are pre-created
in AWS console. This file just references them by name.

Each connector defines:
  - identity_provider: Name of the credential provider in AgentCore Identity
  - mcp_endpoint: Remote MCP server URL (agent connects directly, no gateway)
  - scopes: OAuth scopes to request
  - custom_parameters: Extra params for the OAuth authorization request
  - tools: Available MCP tools grouped by category (read/write) for UI rendering
"""

# Pre-existing workload identity name (created in AWS console)
WORKLOAD_IDENTITY_NAME = "slack-oauth-test-agent"

# Callback URL — leave empty for local testing
CALLBACK_URLS = ["https://dev.qubitz.ai/control-hub/"]

CONNECTORS = {
    "slack": {
        "display_name": "Slack",
        "identity_provider": "slack-provider",
        "mcp_endpoint": "https://mcp.slack.com/mcp",
        "scopes": [
            "chat:write", "channels:history", "groups:history",
            "im:history", "mpim:history",
            "search:read.public", "search:read.private", "search:read.mpim",
            "search:read.im", "search:read.files", "search:read.users",
            "canvases:read", "canvases:write",
            "users:read", "users:read.email",
        ],
        "tools": {
            "read": [
                {"name": "slack_list_channels", "label": "List Channels"},
                {"name": "slack_get_channel_history", "label": "Get Channel History"},
                {"name": "slack_get_thread_replies", "label": "Get Thread Replies"},
                {"name": "slack_get_users", "label": "Get Users"},
                {"name": "slack_get_user_profile", "label": "Get User Profile"},
                {"name": "slack_search_messages", "label": "Search Messages"},
            ],
            "write": [
                {"name": "slack_post_message", "label": "Post Message"},
                {"name": "slack_reply_to_thread", "label": "Reply to Thread"},
                {"name": "slack_add_reaction", "label": "Add Reaction"},
            ],
        },
    },
    "gmail": {
        "display_name": "Gmail",
        "identity_provider": "google-gmail-provider",
        "mcp_endpoint": "https://mcp.qubitz.ai/google/gmail/mcp",
        "scopes": [
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.compose",
        ],
        "custom_parameters": {"access_type": "offline", "prompt": "consent"},
        "tools": {
            "read": [
                {"name": "get_profile", "label": "Get Gmail Profile"},
                {"name": "search_messages", "label": "Search Emails"},
                {"name": "read_message", "label": "Read Email"},
                {"name": "read_thread", "label": "Read Thread"},
                {"name": "list_drafts", "label": "List Drafts"},
            ],
            "write": [
                {"name": "create_draft", "label": "Create Draft"},
                {"name": "send_message", "label": "Send Email"},
                {"name": "reply_to_message", "label": "Reply to Email"},
            ],
        },
    },
    "google_calendar": {
        "display_name": "Google Calendar",
        "identity_provider": "google-calendar-provider",
        "mcp_endpoint": "https://mcp.qubitz.ai/google/calendar/mcp",
        "scopes": [
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.events",
        ],
        "custom_parameters": {"access_type": "offline", "prompt": "consent"},
        "tools": {
            "read": [
                {"name": "list_calendars", "label": "List Calendars"},
                {"name": "list_calendar_events", "label": "List Calendar Events"},
                {"name": "get_event_details", "label": "Get Event Details"},
                {"name": "find_free_time", "label": "Find Free Time"},
                {"name": "find_meeting_times", "label": "Find Meeting Times"},
            ],
            "write": [
                {"name": "create_calendar_event", "label": "Create Calendar Event"},
                {"name": "update_calendar_event", "label": "Update Calendar Event"},
                {"name": "delete_calendar_event", "label": "Delete Calendar Event"},
                {"name": "respond_to_calendar_event", "label": "Respond to Calendar Event"},
            ],
        },
    },
    "google_drive": {
        "display_name": "Google Drive & Sheets",
        "identity_provider": "google-drive-provider",
        "mcp_endpoint": "https://mcp.qubitz.ai/google/drive/mcp",
        "scopes": [
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/documents",
        ],
        "custom_parameters": {"access_type": "offline", "prompt": "consent"},
        "tools": {
            "read": [
                {"name": "search_files", "label": "Search Drive Files"},
                {"name": "list_files", "label": "List Files"},
                {"name": "read_file", "label": "Read File Content"},
                {"name": "get_file_details", "label": "Get File Details"},
            ],
            "write": [
                {"name": "create_file", "label": "Create File"},
                {"name": "share_file", "label": "Share File"},
                {"name": "move_file", "label": "Move File"},
                {"name": "delete_file", "label": "Delete File"},
            ],
        },
    },
    "microsoft": {
        "display_name": "Microsoft Teams",
        "identity_provider": "microsoft-provider",
        "mcp_endpoint": "https://mcp.qubitz.ai/microsoft/teams/mcp",
        "scopes": [
            "openid", "profile", "email", "offline_access",
            "User.Read",
            "Chat.ReadWrite",
            "ChannelMessage.Send",
        ],
        "tools": {
            "read": [
                {"name": "teams_get_profile", "label": "Get Profile"},
                {"name": "teams_list_teams", "label": "List Teams"},
                {"name": "teams_list_channels", "label": "List Channels"},
                {"name": "teams_list_members", "label": "List Members"},
                {"name": "teams_read_messages", "label": "Read Channel Messages"},
                {"name": "teams_read_replies", "label": "Read Thread Replies"},
                {"name": "teams_list_chats", "label": "List Chats"},
                {"name": "teams_read_chat_messages", "label": "Read Chat Messages"},
            ],
            "write": [
                {"name": "teams_send_message", "label": "Send Channel Message"},
                {"name": "teams_reply_to_message", "label": "Reply to Channel Message"},
                {"name": "teams_send_chat_message", "label": "Send Chat Message"},
            ],
        },
    },
}
