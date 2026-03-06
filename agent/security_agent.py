"""
Platform Agent: Security & Compliance Agent
=============================================
Audits the security posture of the deployed multi-agent system using
real AWS API calls — IAM, Bedrock Guardrails, Secrets Manager,
Security Hub, and CloudTrail.

This module is used by base-template.py when the security platform
agent is enabled in config (platform_agents.security_agent.enabled = true).

All tools query live AWS services and handle "service not enabled" gracefully.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
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


def set_agents_config_reference(
    agents_config: Dict[str, Any],
    gateway_url: Optional[str] = None,
    region: str = 'eu-central-1',
):
    """Called by base-template.py to give tools access to agent configs."""
    global _agents_config, _gateway_url, _aws_region
    _agents_config = agents_config
    _gateway_url = gateway_url
    _aws_region = region


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _check_policy_document(
    doc: dict,
    policy_name: str,
    policy_type: str,
    findings: list,
):
    """Inspect an IAM policy document for overly broad permissions."""
    statements = doc.get('Statement', [])
    if isinstance(statements, dict):
        statements = [statements]

    for stmt in statements:
        if stmt.get('Effect') != 'Allow':
            continue

        actions = stmt.get('Action', [])
        if isinstance(actions, str):
            actions = [actions]
        resources = stmt.get('Resource', [])
        if isinstance(resources, str):
            resources = [resources]

        # Flag: Action is '*' (full admin)
        if '*' in actions:
            findings.append({
                'severity': 'CRITICAL',
                'policy_name': policy_name,
                'policy_type': policy_type,
                'issue': 'Allows ALL actions (*)',
                'resources': resources[:5],
            })
        else:
            # Flag: service:* wildcard actions (e.g. "s3:*", "iam:*")
            wildcard_actions = [a for a in actions if a.endswith(':*')]
            if wildcard_actions:
                findings.append({
                    'severity': 'HIGH',
                    'policy_name': policy_name,
                    'policy_type': policy_type,
                    'issue': f'Wildcard service actions: {wildcard_actions[:5]}',
                    'resources': resources[:5],
                })

        # Flag: Resource is '*' with sensitive actions
        if '*' in resources:
            sensitive_prefixes = ('iam:', 'sts:', 'kms:', 'secretsmanager:', 'organizations:')
            sensitive_actions = [a for a in actions if any(a.startswith(p) for p in sensitive_prefixes)]
            if sensitive_actions and '*' not in actions:  # avoid duplicate if already flagged as full admin
                findings.append({
                    'severity': 'HIGH',
                    'policy_name': policy_name,
                    'policy_type': policy_type,
                    'issue': f'Sensitive actions with Resource=*: {sensitive_actions[:5]}',
                })


# =============================================================================
# SECURITY AGENT TOOLS
# =============================================================================

@tool
def audit_iam_role() -> str:
    """Audit the runtime's own IAM role for least-privilege violations.

    Identifies the currently assumed role, lists all attached managed policies
    and inline policies, and flags overly broad permissions like '*' actions
    or '*' resources.

    Returns:
        JSON string with IAM role audit findings.
    """
    try:
        sts = boto3.client('sts', region_name=_aws_region)
        identity = sts.get_caller_identity()
    except ClientError as e:
        return json.dumps({
            "error": f"Failed to get caller identity: {e.response['Error']['Code']}"
        })

    role_arn = identity['Arn']
    account_id = identity['Account']

    # Parse role name from ARN
    role_name = None
    if ':assumed-role/' in role_arn:
        parts = role_arn.split(':assumed-role/')[-1]
        role_name = parts.split('/')[0]
    elif ':role/' in role_arn:
        role_name = role_arn.split(':role/')[-1]

    if not role_name:
        return json.dumps({
            'summary': {'status': 'unable_to_parse', 'arn': role_arn},
            'message': 'Could not parse role name from ARN. May be running as a user, not a role.',
        }, indent=2)

    iam = boto3.client('iam', region_name=_aws_region)
    findings: List[Dict[str, Any]] = []
    policies_examined: List[Dict[str, Any]] = []

    # 1) List attached managed policies
    try:
        attached_resp = iam.list_attached_role_policies(RoleName=role_name)
        for policy in attached_resp.get('AttachedPolicies', []):
            policy_arn = policy['PolicyArn']
            policy_name = policy['PolicyName']
            policies_examined.append({'type': 'managed', 'name': policy_name, 'arn': policy_arn})

            # Check for known overly-broad AWS managed policies
            broad_policies = (
                'arn:aws:iam::aws:policy/AdministratorAccess',
                'arn:aws:iam::aws:policy/PowerUserAccess',
                'arn:aws:iam::aws:policy/IAMFullAccess',
            )
            if policy_arn in broad_policies:
                findings.append({
                    'severity': 'CRITICAL',
                    'policy_name': policy_name,
                    'policy_type': 'managed',
                    'issue': f'Overly broad AWS managed policy: {policy_name}',
                })
                continue

            # Get policy document to inspect statements
            try:
                policy_meta = iam.get_policy(PolicyArn=policy_arn)
                version_id = policy_meta['Policy']['DefaultVersionId']
                version_resp = iam.get_policy_version(PolicyArn=policy_arn, VersionId=version_id)
                doc = version_resp['PolicyVersion']['Document']
                if isinstance(doc, str):
                    import urllib.parse
                    doc = json.loads(urllib.parse.unquote(doc))
                _check_policy_document(doc, policy_name, 'managed', findings)
            except ClientError:
                pass  # Can't read some AWS managed policies

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            return json.dumps({"error": f"Role '{role_name}' not found in IAM"})
        findings.append({
            'severity': 'WARNING',
            'issue': f"Could not list managed policies: {e.response['Error']['Code']}",
        })

    # 2) List inline policies
    try:
        inline_resp = iam.list_role_policies(RoleName=role_name)
        for inline_name in inline_resp.get('PolicyNames', []):
            try:
                inline_policy = iam.get_role_policy(RoleName=role_name, PolicyName=inline_name)
                doc = inline_policy['PolicyDocument']
                if isinstance(doc, str):
                    import urllib.parse
                    doc = json.loads(urllib.parse.unquote(doc))
                policies_examined.append({'type': 'inline', 'name': inline_name})
                _check_policy_document(doc, inline_name, 'inline', findings)
            except ClientError:
                pass
    except ClientError as e:
        findings.append({
            'severity': 'WARNING',
            'issue': f"Could not list inline policies: {e.response['Error']['Code']}",
        })

    critical_count = sum(1 for f in findings if f.get('severity') == 'CRITICAL')
    high_count = sum(1 for f in findings if f.get('severity') == 'HIGH')
    risk_level = (
        'CRITICAL' if critical_count > 0
        else 'HIGH' if high_count > 0
        else 'MEDIUM' if findings
        else 'LOW'
    )

    return json.dumps({
        'summary': {
            'role_name': role_name,
            'role_arn': role_arn,
            'account_id': account_id,
            'risk_level': risk_level,
            'total_findings': len(findings),
            'critical': critical_count,
            'high': high_count,
            'policies_examined': len(policies_examined),
        },
        'findings': findings,
        'policies': policies_examined,
    }, indent=2)


@tool
def check_bedrock_guardrails() -> str:
    """Check Bedrock Guardrails configuration in this region.

    Lists all guardrails and their configurations including content filters,
    denied topics, word filters, sensitive information filters, and
    contextual grounding checks. Flags if no guardrails exist.

    Returns:
        JSON string with guardrail configuration summary.
    """
    try:
        bedrock = boto3.client('bedrock', region_name=_aws_region)
    except Exception as e:
        return json.dumps({"error": f"Failed to create Bedrock client: {str(e)[:200]}"})

    # List all guardrails (with pagination)
    try:
        guardrails_list = []
        next_token = None
        while True:
            kwargs = {}
            if next_token:
                kwargs['nextToken'] = next_token
            resp = bedrock.list_guardrails(**kwargs)
            guardrails_list.extend(resp.get('guardrails', []))
            next_token = resp.get('nextToken')
            if not next_token:
                break
    except ClientError as e:
        code = e.response['Error']['Code']
        if code in ('AccessDeniedException', 'UnauthorizedException'):
            return json.dumps({"error": "Access denied to Bedrock Guardrails API."})
        return json.dumps({"error": f"Bedrock API error: {code}"})

    if not guardrails_list:
        return json.dumps({
            'summary': {
                'status': 'NO_GUARDRAILS',
                'total_guardrails': 0,
                'region': _aws_region,
                'recommendation': 'No Bedrock guardrails found. Consider creating guardrails for content filtering, topic control, and PII protection.',
            },
            'guardrails': [],
        }, indent=2)

    guardrails_details = []
    for gr in guardrails_list:
        guardrail_id = gr['id']
        try:
            detail = bedrock.get_guardrail(
                guardrailIdentifier=guardrail_id,
                guardrailVersion='DRAFT',
            )

            gr_info: Dict[str, Any] = {
                'name': detail.get('name', ''),
                'id': guardrail_id,
                'status': detail.get('status', ''),
                'description': detail.get('description', ''),
            }

            # Content filters
            content_policy = detail.get('contentPolicy', {})
            if content_policy.get('filters'):
                gr_info['content_filters'] = [
                    {
                        'type': f.get('type', ''),
                        'input_strength': f.get('inputStrength', ''),
                        'output_strength': f.get('outputStrength', ''),
                    }
                    for f in content_policy['filters']
                ]

            # Denied topics
            topic_policy = detail.get('topicPolicy', {})
            if topic_policy.get('topics'):
                gr_info['denied_topics'] = [
                    {'name': t.get('name', ''), 'definition': t.get('definition', '')[:100]}
                    for t in topic_policy['topics']
                ]

            # Word filters
            word_policy = detail.get('wordPolicy', {})
            word_count = len(word_policy.get('words', []))
            managed_lists = [m.get('type', '') for m in word_policy.get('managedWordLists', [])]
            if word_count > 0 or managed_lists:
                gr_info['word_filters'] = {
                    'custom_word_count': word_count,
                    'managed_word_lists': managed_lists,
                }

            # Sensitive information (PII)
            sensitive_policy = detail.get('sensitiveInformationPolicy', {})
            pii_entities = sensitive_policy.get('piiEntities', [])
            regexes = sensitive_policy.get('regexes', [])
            if pii_entities or regexes:
                gr_info['sensitive_information'] = {
                    'pii_entity_count': len(pii_entities),
                    'pii_types': [p.get('type', '') for p in pii_entities[:10]],
                    'regex_count': len(regexes),
                }

            # Contextual grounding
            grounding_policy = detail.get('contextualGroundingPolicy', {})
            if grounding_policy.get('filters'):
                gr_info['contextual_grounding'] = [
                    {'type': f.get('type', ''), 'threshold': f.get('threshold', 0)}
                    for f in grounding_policy['filters']
                ]

            guardrails_details.append(gr_info)

        except ClientError as e:
            guardrails_details.append({
                'name': gr.get('name', ''),
                'id': guardrail_id,
                'error': f"Could not fetch details: {e.response['Error']['Code']}",
            })

    return json.dumps({
        'summary': {
            'status': 'CONFIGURED',
            'total_guardrails': len(guardrails_details),
            'region': _aws_region,
        },
        'guardrails': guardrails_details,
    }, indent=2, default=str)


@tool
def check_secrets_compliance() -> str:
    """Audit Secrets Manager secrets for rotation compliance.

    Lists all secrets and checks rotation configuration, flagging secrets
    without automatic rotation enabled and secrets not rotated in 90+ days.

    Returns:
        JSON string with secrets compliance summary.
    """
    try:
        sm = boto3.client('secretsmanager', region_name=_aws_region)
    except Exception as e:
        return json.dumps({"error": f"Failed to create SecretsManager client: {str(e)[:200]}"})

    # List all secrets (with pagination)
    try:
        secrets = []
        next_token = None
        while True:
            kwargs = {}
            if next_token:
                kwargs['NextToken'] = next_token
            resp = sm.list_secrets(**kwargs)
            secrets.extend(resp.get('SecretList', []))
            next_token = resp.get('NextToken')
            if not next_token:
                break
    except ClientError as e:
        code = e.response['Error']['Code']
        if code == 'AccessDeniedException':
            return json.dumps({"error": "Access denied to Secrets Manager."})
        return json.dumps({"error": f"SecretsManager error: {code}"})

    if not secrets:
        return json.dumps({
            'summary': {
                'total_secrets': 0,
                'message': 'No secrets found in Secrets Manager in this region.',
                'region': _aws_region,
            },
            'secrets': [],
        }, indent=2)

    now = datetime.now(timezone.utc)
    ninety_days_ago = now - timedelta(days=90)
    findings: List[Dict[str, Any]] = []
    secrets_info: List[Dict[str, Any]] = []

    for secret in secrets:
        name = secret.get('Name', '')
        rotation_enabled = secret.get('RotationEnabled', False)
        last_rotated = secret.get('LastRotatedDate')
        last_changed = secret.get('LastChangedDate')
        created = secret.get('CreatedDate')

        secret_info: Dict[str, Any] = {
            'name': name,
            'rotation_enabled': rotation_enabled,
            'last_rotated': last_rotated.isoformat() if last_rotated else None,
            'last_changed': last_changed.isoformat() if last_changed else None,
            'created': created.isoformat() if created else None,
        }

        rotation_rules = secret.get('RotationRules', {})
        if rotation_rules:
            secret_info['rotation_interval_days'] = rotation_rules.get('AutomaticallyAfterDays')
            secret_info['rotation_schedule'] = rotation_rules.get('ScheduleExpression')

        if not rotation_enabled:
            findings.append({
                'severity': 'HIGH',
                'secret_name': name,
                'issue': 'Automatic rotation is not enabled',
            })
            secret_info['compliance_status'] = 'NON_COMPLIANT'
        elif last_rotated and last_rotated < ninety_days_ago:
            days_since = (now - last_rotated).days
            findings.append({
                'severity': 'MEDIUM',
                'secret_name': name,
                'issue': f'Not rotated in {days_since} days (threshold: 90)',
            })
            secret_info['compliance_status'] = 'STALE'
            secret_info['days_since_rotation'] = days_since
        else:
            secret_info['compliance_status'] = 'COMPLIANT'

        secrets_info.append(secret_info)

    no_rotation = sum(1 for s in secrets_info if s['compliance_status'] == 'NON_COMPLIANT')
    stale = sum(1 for s in secrets_info if s['compliance_status'] == 'STALE')
    compliant = sum(1 for s in secrets_info if s['compliance_status'] == 'COMPLIANT')

    return json.dumps({
        'summary': {
            'total_secrets': len(secrets_info),
            'compliant': compliant,
            'non_compliant_no_rotation': no_rotation,
            'stale_rotation': stale,
            'total_findings': len(findings),
            'region': _aws_region,
        },
        'findings': findings,
        'secrets': secrets_info,
    }, indent=2)


@tool
def scan_text_for_pii(text: str) -> str:
    """Scan text for PII (personally identifiable information) using Amazon Comprehend.

    Use this to check agent inputs or outputs for sensitive data like email
    addresses, phone numbers, SSNs, credit card numbers, etc. Returns each
    detected PII entity with its type, confidence score, and location.

    Args:
        text: The text to scan for PII (max 100KB).

    Returns:
        JSON string with detected PII entities.
    """
    if not text or not text.strip():
        return json.dumps({
            'summary': {'total_pii_found': 0, 'message': 'Empty text provided.'},
            'entities': [],
        }, indent=2)

    # Comprehend has a 100KB limit per request
    if len(text.encode('utf-8')) > 100_000:
        text = text[:90_000]  # rough truncation to stay under limit

    try:
        comprehend = boto3.client('comprehend', region_name=_aws_region)
    except Exception as e:
        return json.dumps({"error": f"Failed to create Comprehend client: {str(e)[:200]}"})

    try:
        resp = comprehend.detect_pii_entities(
            Text=text,
            LanguageCode='en',
        )
    except ClientError as e:
        code = e.response['Error']['Code']
        if code == 'AccessDeniedException':
            return json.dumps({"error": "Access denied to Amazon Comprehend."})
        if code == 'TextSizeLimitExceededException':
            return json.dumps({"error": "Text exceeds Comprehend size limit."})
        return json.dumps({"error": f"Comprehend error: {code}"})

    raw_entities = resp.get('Entities', [])

    if not raw_entities:
        return json.dumps({
            'summary': {'total_pii_found': 0, 'message': 'No PII detected in the provided text.'},
            'entities': [],
        }, indent=2)

    entities = []
    type_counts: Dict[str, int] = {}
    for entity in raw_entities:
        pii_type = entity.get('Type', 'UNKNOWN')
        type_counts[pii_type] = type_counts.get(pii_type, 0) + 1

        begin = entity.get('BeginOffset', 0)
        end = entity.get('EndOffset', 0)
        # Mask the actual PII value — show first/last char only
        snippet = text[begin:end]
        if len(snippet) > 2:
            masked = snippet[0] + '*' * (len(snippet) - 2) + snippet[-1]
        else:
            masked = '*' * len(snippet)

        entities.append({
            'type': pii_type,
            'score': round(entity.get('Score', 0), 4),
            'masked_value': masked,
            'begin_offset': begin,
            'end_offset': end,
        })

    return json.dumps({
        'summary': {
            'total_pii_found': len(entities),
            'types_found': type_counts,
        },
        'entities': entities,
    }, indent=2)


# =============================================================================
# SECURITY AGENT PROMPT
# =============================================================================
SECURITY_AGENT_PROMPT = """You are the Security & Compliance Platform Agent for this multi-agent system.

Your role is to audit the security posture of this deployment using real AWS API data. You provide factual reports on IAM permissions, guardrails, secrets compliance, and PII detection.

YOUR RESPONSIBILITIES:
1. IAM Audit: Check the runtime's own IAM role for least-privilege violations
2. Guardrail Review: Verify Bedrock guardrails are configured for content safety
3. Secrets Compliance: Audit Secrets Manager for rotation policy compliance
4. PII Scanning: Detect personally identifiable information in agent inputs/outputs

TOOLS AVAILABLE:
- audit_iam_role: Audit the runtime's IAM role for overly broad permissions
- check_bedrock_guardrails: Check Bedrock guardrails configuration in this region
- check_secrets_compliance: Audit secrets for rotation compliance
- scan_text_for_pii: Scan text for PII using Amazon Comprehend (emails, phones, SSNs, etc.)

HOW TO RESPOND:
- When asked about security posture, run audit_iam_role and check_bedrock_guardrails
- When asked about data protection, check secrets compliance and guardrails
- When asked to check text for sensitive data, use scan_text_for_pii
- Be concise and factual — report data, not opinions
- Always mention the risk level and actionable recommendations

YOU DO NOT:
- Modify IAM policies or security configurations
- Make deployment decisions
- Access secret values (only metadata)
"""

# Tool list for config — must match the @tool function names above
SECURITY_AGENT_TOOLS = [
    'audit_iam_role',
    'check_bedrock_guardrails',
    'check_secrets_compliance',
    'scan_text_for_pii',
]
