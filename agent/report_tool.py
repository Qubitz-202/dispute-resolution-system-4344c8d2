"""
Report Generation Tool
======================
Converts markdown content to a styled PDF and uploads to S3.
The PDF is publicly accessible via an S3 URL.

Uses fpdf2 (pure Python, no system deps, ARM64 compatible) so it works
in AgentCore's managed PYTHON_3_12 runtime without custom Docker.
"""

import os
import json
import logging
import tempfile
import re
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

try:
    from strands import tool
except ImportError:
    def tool(fn):
        fn.fn = fn
        return fn

logger = logging.getLogger(__name__)

S3_BUCKET = 'multiagent-user-reports'
AWS_REGION = os.environ.get('AWS_REGION', 'eu-central-1')


def _md_to_pdf(md_content: str, title: str, output_path: str) -> bool:
    """Convert markdown content directly to PDF using fpdf2 (lazy imported)."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # Title
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        # Title underline
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
        pdf.ln(8)

        # Parse markdown line by line
        lines = md_content.split('\n')
        in_code_block = False
        in_table = False
        table_rows = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    in_code_block = False
                    pdf.ln(4)
                else:
                    in_code_block = True
                    pdf.ln(2)
                i += 1
                continue

            if in_code_block:
                pdf.set_font('Courier', '', 9)
                pdf.set_fill_color(244, 244, 244)
                pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT", fill=True)
                i += 1
                continue

            # Table rows
            if '|' in line and line.strip().startswith('|'):
                # Skip separator rows like |---|---|
                if re.match(r'^\s*\|[\s\-:|]+\|\s*$', line):
                    i += 1
                    continue
                cells = [c.strip() for c in line.strip().strip('|').split('|')]
                if not in_table:
                    in_table = True
                    table_rows = []
                table_rows.append(cells)
                # Check if next line is still a table
                if i + 1 >= len(lines) or '|' not in lines[i + 1] or not lines[i + 1].strip().startswith('|'):
                    _render_table(pdf, table_rows)
                    in_table = False
                    table_rows = []
                i += 1
                continue

            stripped = line.strip()

            # Empty line
            if not stripped:
                pdf.ln(4)
                i += 1
                continue

            # Horizontal rule
            if re.match(r'^[-*_]{3,}\s*$', stripped):
                pdf.ln(4)
                pdf.set_draw_color(200, 200, 200)
                pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
                pdf.ln(4)
                i += 1
                continue

            # Headers
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                text = stripped.lstrip('#').strip()
                if level == 1:
                    pdf.ln(8)
                    pdf.set_font('Helvetica', 'B', 16)
                elif level == 2:
                    pdf.ln(10)
                    pdf.set_font('Helvetica', 'B', 14)
                elif level == 3:
                    pdf.ln(6)
                    pdf.set_font('Helvetica', 'B', 12)
                else:
                    pdf.ln(4)
                    pdf.set_font('Helvetica', 'B', 11)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 7, text)
                if level == 2:
                    pdf.set_draw_color(220, 220, 220)
                    pdf.line(10, pdf.get_y() + 1, pdf.w - 10, pdf.get_y() + 1)
                    pdf.ln(3)
                else:
                    pdf.ln(2)
                i += 1
                continue

            # Blockquote
            if stripped.startswith('>'):
                text = stripped.lstrip('>').strip()
                pdf.set_font('Helvetica', 'I', 10)
                pdf.set_text_color(80, 80, 80)
                pdf.set_fill_color(249, 249, 249)
                x = pdf.get_x()
                pdf.set_x(x + 5)
                pdf.multi_cell(pdf.w - 25, 6, text, fill=True)
                pdf.set_x(x)
                pdf.ln(2)
                i += 1
                continue

            # Bullet lists
            if re.match(r'^[\-\*\+]\s', stripped):
                text = re.sub(r'^[\-\*\+]\s', '', stripped)
                pdf.set_font('Helvetica', '', 11)
                pdf.set_text_color(51, 51, 51)
                x = pdf.get_x()
                pdf.set_x(x + 5)
                pdf.cell(5, 6, '\u2022')
                pdf.multi_cell(pdf.w - 30, 6, _strip_md_inline(text))
                pdf.set_x(x)
                i += 1
                continue

            # Numbered lists
            m = re.match(r'^(\d+)\.\s', stripped)
            if m:
                num = m.group(1)
                text = re.sub(r'^\d+\.\s', '', stripped)
                pdf.set_font('Helvetica', '', 11)
                pdf.set_text_color(51, 51, 51)
                x = pdf.get_x()
                pdf.set_x(x + 5)
                pdf.cell(8, 6, f'{num}.')
                pdf.multi_cell(pdf.w - 33, 6, _strip_md_inline(text))
                pdf.set_x(x)
                i += 1
                continue

            # Normal paragraph
            pdf.set_font('Helvetica', '', 11)
            pdf.set_text_color(51, 51, 51)
            pdf.multi_cell(0, 6, _strip_md_inline(stripped))
            pdf.ln(2)
            i += 1

        pdf.output(output_path)
        return True
    except ImportError:
        logger.error("fpdf2 not installed")
        return False
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return False


def _strip_md_inline(text: str) -> str:
    """Remove inline markdown formatting (bold, italic, code, links)."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    return text


def _render_table(pdf, rows: list):
    """Render a markdown table into the PDF."""
    if not rows:
        return

    num_cols = max(len(r) for r in rows)
    col_width = (pdf.w - 20) / max(num_cols, 1)

    for row_idx, row in enumerate(rows):
        # Pad row if fewer cells
        while len(row) < num_cols:
            row.append('')

        if row_idx == 0:
            # Header row
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_fill_color(242, 242, 242)
            pdf.set_text_color(0, 0, 0)
            for cell in row:
                pdf.cell(col_width, 7, _strip_md_inline(cell)[:50], border=1, fill=True)
            pdf.ln()
        else:
            pdf.set_font('Helvetica', '', 10)
            pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(51, 51, 51)
            for cell in row:
                pdf.cell(col_width, 7, _strip_md_inline(cell)[:50], border=1)
            pdf.ln()

    pdf.ln(4)


def _ensure_bucket(s3_client) -> bool:
    """Create the S3 bucket if it doesn't exist and configure for public read."""
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            try:
                s3_client.create_bucket(
                    Bucket=S3_BUCKET,
                    CreateBucketConfiguration={'LocationConstraint': AWS_REGION},
                )
                # Disable block public access
                s3_client.put_public_access_block(
                    Bucket=S3_BUCKET,
                    PublicAccessBlockConfiguration={
                        'BlockPublicAcls': False,
                        'IgnorePublicAcls': False,
                        'BlockPublicPolicy': False,
                        'RestrictPublicBuckets': False,
                    },
                )
                # Set bucket policy for public read
                policy = {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Sid": "PublicRead",
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{S3_BUCKET}/*",
                    }],
                }
                s3_client.put_bucket_policy(
                    Bucket=S3_BUCKET,
                    Policy=json.dumps(policy),
                )
                logger.info(f"Created public S3 bucket: {S3_BUCKET}")
                return True
            except Exception as create_err:
                logger.error(f"Failed to create bucket: {create_err}")
                return False
        else:
            logger.error(f"Bucket check failed: {e}")
            return False


def _slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text[:80]


@tool
def generate_report(title: str, content: str) -> str:
    """Generate a styled PDF report from markdown content and upload to S3.

    Creates a professional PDF from the provided markdown content with
    consistent styling (tables, headers, code blocks supported). The PDF
    is uploaded to a public S3 bucket and a download URL is returned.

    IMPORTANT: Only call this tool when the user EXPLICITLY asks to
    generate or create a report/PDF. Do NOT call this automatically.

    Args:
        title: Report title (used in filename and as the PDF heading).
        content: Full markdown content for the report body.

    Returns:
        JSON string with the public S3 URL for the PDF, or an error message.
    """
    user_id = os.environ.get('USER_ID', 'unknown')
    project_id = os.environ.get('PROJECT_ID', 'unknown')
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    slug = _slugify(title) or 'report'
    filename = f'{slug}_{timestamp}.pdf'
    s3_key = f'{user_id}/{project_id}/pdfs/{filename}'

    # Convert to PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        success = _md_to_pdf(content, title, tmp_path)
        if not success:
            return json.dumps({
                'error': 'pdf_conversion_failed',
                'message': 'Failed to convert markdown to PDF. Check logs for details.',
            })

        # Check file size
        file_size = os.path.getsize(tmp_path)
        if file_size == 0:
            return json.dumps({
                'error': 'empty_pdf',
                'message': 'PDF conversion produced an empty file.',
            })

        # Upload to S3
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        _ensure_bucket(s3_client)

        s3_client.upload_file(
            tmp_path, S3_BUCKET, s3_key,
            ExtraArgs={'ContentType': 'application/pdf'},
        )

        public_url = f'https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}'

        return json.dumps({
            'status': 'success',
            'url': public_url,
            's3_bucket': S3_BUCKET,
            's3_key': s3_key,
            'filename': filename,
            'file_size_bytes': file_size,
            'title': title,
        }, indent=2)

    except ClientError as e:
        code = e.response['Error']['Code']
        msg = e.response['Error'].get('Message', str(e))
        logger.error(f"S3 upload failed: {code} — {msg}")
        return json.dumps({'error': code, 'message': msg[:300]})
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return json.dumps({'error': 'unexpected', 'message': str(e)[:300]})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
