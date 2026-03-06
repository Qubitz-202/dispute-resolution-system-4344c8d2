# =============================================================================
# IMAGE GENERATION AND ANALYSIS TOOLS
# Using Stability AI Stable Diffusion & Claude Vision on AWS Bedrock
# =============================================================================

import os
import json
import base64
import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
from strands import tool
from PIL import Image
import io

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
# S3 Configuration for image storage
S3_IMAGES_BUCKET = os.environ.get('S3_IMAGES_BUCKET', 'qubitz-customer-prod-v2')
USER_ID = os.environ.get('USER_ID', '202')
PROJECT_ID = os.environ.get('PROJECT_ID', '202')
IMAGES_PREFIX = f"{USER_ID}/{PROJECT_ID}/images/"

# Bedrock Configuration
BEDROCK_REGION = os.environ.get('BEDROCK_REGION', 'us-west-2')

# Available Stability AI models on Bedrock
AVAILABLE_IMAGE_MODELS = {
    'sd3-large': 'stability.sd3-5-large-v1:0',  # Stable Diffusion 3 Large
    'stable-image-ultra': 'stability.stable-image-ultra-v1:0',  # Highest quality
    'stable-image-core': 'stability.stable-image-core-v1:0',  # Fast & efficient
    'sdxl': 'stability.stable-diffusion-xl-v1'  # SDXL (being deprecated)
}

# Default model
DEFAULT_IMAGE_MODEL = 'sd3-large'

# Vision model for image analysis
VISION_MODEL = os.environ.get('VISION_MODEL', 'anthropic.claude-3-5-sonnet-20241022-v2:0')

# AWS Clients
s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'eu-central-1'))
bedrock_runtime = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Valid aspect ratios for SD3 Large and Stable Image models
VALID_ASPECT_RATIOS = ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]

def _get_aspect_ratio(width: int, height: int) -> str:
    """
    Map width/height dimensions to the closest valid aspect ratio.
    
    Valid ratios: 1:1, 16:9, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21
    """
    if width == height:
        return "1:1"
    
    ratio = width / height
    
    # Map common ratios
    ratio_map = {
        1.0: "1:1",
        16/9: "16:9",
        21/9: "21:9",
        2/3: "2:3",
        3/2: "3:2",
        4/5: "4:5",
        5/4: "5:4",
        9/16: "9:16",
        9/21: "9:21"
    }
    
    # Find closest matching ratio
    closest_ratio = min(ratio_map.keys(), key=lambda x: abs(x - ratio))
    return ratio_map[closest_ratio]


# =============================================================================
# IMAGE GENERATION TOOL
# =============================================================================
@tool
def generate_image(
    prompt: str,
    model: str = DEFAULT_IMAGE_MODEL,
    negative_prompt: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None
) -> str:
    """
    Generate an image using Stability AI models on AWS Bedrock and store it in S3.
    
    Args:
        prompt: Text description of the image to generate (required)
        model: Model to use - options: 'sd3-large' (default), 'stable-image-ultra', 
               'stable-image-core', 'sdxl'
        negative_prompt: What you DON'T want in the image (optional)
        width: Image width in pixels (default: 1024)
        height: Image height in pixels (default: 1024)
        seed: Random seed for reproducibility (optional, auto-generated if not provided)
    
    Returns:
        JSON string containing:
        - s3_uri: S3 URI of the generated image
        - s3_url: HTTPS URL to access the image
        - image_key: S3 object key
        - metadata: Generation details (prompt, model, dimensions, seed)
    
    Example:
        generate_image("A cute robot in a futuristic city")
        generate_image("Mountain landscape at sunset", model="stable-image-ultra")
    """
    try:
        # Validate model
        if model not in AVAILABLE_IMAGE_MODELS:
            return json.dumps({
                "error": f"Invalid model '{model}'. Available models: {list(AVAILABLE_IMAGE_MODELS.keys())}",
                "status": "failed"
            })
        
        model_id = AVAILABLE_IMAGE_MODELS[model]
        
        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 4294967295)
        
        logger.info(f"Generating image with model: {model_id}")
        logger.info(f"Prompt: {prompt}")
        
        # Prepare request based on model type
        if model == 'sdxl':
            # SDXL uses different format
            request_body = {
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 10,
                "seed": seed,
                "steps": 30,
                "width": width,
                "height": height
            }
            if negative_prompt:
                request_body["text_prompts"].append({
                    "text": negative_prompt,
                    "weight": -1.0
                })
        else:
            # SD3 Large, Stable Image Ultra/Core use newer format
            # Map dimensions to valid aspect ratios
            aspect_ratio = _get_aspect_ratio(width, height)
            
            request_body = {
                "prompt": prompt,
                "mode": "text-to-image",
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
                "seed": seed
            }
            if negative_prompt:
                request_body["negative_prompt"] = negative_prompt
        
        # Invoke Bedrock model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        # Extract base64 image from response
        if model == 'sdxl':
            # SDXL format
            if 'artifacts' not in response_body or not response_body['artifacts']:
                raise ValueError("No image generated in response")
            
            base64_image = response_body['artifacts'][0]['base64']
            finish_reason = response_body['artifacts'][0].get('finishReason', 'SUCCESS')
            
            if finish_reason == 'ERROR':
                raise ValueError("Image generation failed with ERROR")
            elif finish_reason == 'CONTENT_FILTERED':
                return json.dumps({
                    "error": "Content was filtered. Please try a different prompt.",
                    "status": "content_filtered"
                })
        else:
            # SD3 Large, Stable Image Ultra/Core format
            if 'images' not in response_body or not response_body['images']:
                raise ValueError("No image generated in response")
            
            base64_image = response_body['images'][0]
            finish_reasons = response_body.get('finish_reasons', [None])
            
            if finish_reasons[0] is not None:
                # Content was filtered or error occurred
                return json.dumps({
                    "error": f"Image generation issue: {finish_reasons[0]}",
                    "status": "filtered_or_error"
                })
        
        # Decode base64 image
        image_bytes = base64.b64decode(base64_image)
        
        # Generate S3 key with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        image_filename = f"generated_{timestamp}_{seed}.png"
        image_key = f"{IMAGES_PREFIX}{image_filename}"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_IMAGES_BUCKET,
            Key=image_key,
            Body=image_bytes,
            ContentType='image/png',
            Metadata={
                'prompt': prompt[:1024],  # S3 metadata has size limits
                'model': model,
                'seed': str(seed),
                'dimensions': f"{width}x{height}",
                'generated_at': timestamp
            }
        )
        
        # Generate S3 URI and URL
        s3_uri = f"s3://{S3_IMAGES_BUCKET}/{image_key}"
        s3_url = f"https://{S3_IMAGES_BUCKET}.s3.amazonaws.com/{image_key}"
        
        logger.info(f"✓ Image generated and stored: {s3_uri}")
        
        return json.dumps({
            "status": "success",
            "s3_uri": s3_uri,
            "s3_url": s3_url,
            "image_key": image_key,
            "bucket": S3_IMAGES_BUCKET,
            "metadata": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model": model,
                "model_id": model_id,
                "seed": seed,
                "dimensions": f"{width}x{height}",
                "generated_at": timestamp
            }
        })
    
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return json.dumps({
            "error": str(e),
            "status": "failed"
        })


# =============================================================================
# IMAGE ANALYSIS TOOL (Using Claude Vision)
# =============================================================================
@tool
def analyze_image(
    image_source: str,
    question: str = "What's in this image? Describe it in detail.",
    max_tokens: int = 2048
) -> str:
    """
    Analyze an image using Claude's vision capabilities on AWS Bedrock.
    
    Args:
        image_source: Either:
            - S3 URI (e.g., "s3://bucket/path/image.png")
            - S3 key (e.g., "user/project/images/image.png")
            - Local file path (e.g., "./image.png")
        question: Question to ask about the image (default: general description)
        max_tokens: Maximum tokens in response (default: 2048)
    
    Returns:
        JSON string containing:
        - analysis: Claude's analysis of the image
        - question: The question asked
        - model: Vision model used
        - status: success/failed
    
    Example:
        analyze_image("s3://bucket/image.png", "What objects are in this image?")
        analyze_image("images/photo.jpg", "Is there a person in this image?")
    """
    try:
        # Parse image source and load image bytes
        image_bytes = None
        image_info = ""
        
        if image_source.startswith('s3://'):
            # S3 URI format
            parts = image_source.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            
            logger.info(f"Loading image from S3: {bucket}/{key}")
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_bytes = response['Body'].read()
            image_info = f"s3://{bucket}/{key}"
            
        elif '/' in image_source and not image_source.startswith('./'):
            # Assume S3 key in default bucket
            logger.info(f"Loading image from S3: {S3_IMAGES_BUCKET}/{image_source}")
            response = s3_client.get_object(Bucket=S3_IMAGES_BUCKET, Key=image_source)
            image_bytes = response['Body'].read()
            image_info = f"s3://{S3_IMAGES_BUCKET}/{image_source}"
            
        else:
            # Local file path
            logger.info(f"Loading image from local file: {image_source}")
            with open(image_source, 'rb') as f:
                image_bytes = f.read()
            image_info = image_source
        
        if not image_bytes:
            raise ValueError("Failed to load image bytes")
        
        # Detect image format
        image = Image.open(io.BytesIO(image_bytes))
        image_format = image.format.lower() if image.format else 'png'
        
        # Map PIL format to MIME type
        format_mapping = {
            'jpeg': 'image/jpeg',
            'jpg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        media_type = format_mapping.get(image_format, 'image/png')
        
        logger.info(f"Image format: {image_format}, MIME type: {media_type}")
        
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare request for Claude
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ]
        }
        
        logger.info(f"Analyzing image with {VISION_MODEL}")
        logger.info(f"Question: {question}")
        
        # Invoke Claude vision model
        response = bedrock_runtime.invoke_model(
            modelId=VISION_MODEL,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        # Extract text from response
        if 'content' in response_body and response_body['content']:
            analysis = response_body['content'][0]['text']
        else:
            raise ValueError("No analysis returned from model")
        
        logger.info(f"✓ Image analysis complete")
        
        return json.dumps({
            "status": "success",
            "analysis": analysis,
            "question": question,
            "image_source": image_info,
            "model": VISION_MODEL,
            "tokens_used": response_body.get('usage', {})
        })
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return json.dumps({
            "error": str(e),
            "status": "failed"
        })


# =============================================================================
# LIST GENERATED IMAGES
# =============================================================================
@tool
def list_generated_images(limit: int = 20) -> str:
    """
    List all generated images in the project's S3 folder.
    
    Args:
        limit: Maximum number of images to return (default: 20)
    
    Returns:
        JSON string containing list of generated images with metadata
    
    Example:
        list_generated_images()
        list_generated_images(limit=50)
    """
    try:
        logger.info(f"Listing images from s3://{S3_IMAGES_BUCKET}/{IMAGES_PREFIX}")
        
        response = s3_client.list_objects_v2(
            Bucket=S3_IMAGES_BUCKET,
            Prefix=IMAGES_PREFIX,
            MaxKeys=limit
        )
        
        images = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            
            # Skip the prefix itself
            if key == IMAGES_PREFIX:
                continue
            
            # Get object metadata
            try:
                metadata_response = s3_client.head_object(
                    Bucket=S3_IMAGES_BUCKET,
                    Key=key
                )
                metadata = metadata_response.get('Metadata', {})
            except:
                metadata = {}
            
            images.append({
                'filename': key.replace(IMAGES_PREFIX, ''),
                'key': key,
                's3_uri': f"s3://{S3_IMAGES_BUCKET}/{key}",
                's3_url': f"https://{S3_IMAGES_BUCKET}.s3.amazonaws.com/{key}",
                'size': obj['Size'],
                'last_modified': obj['LastModified'].isoformat(),
                'metadata': metadata
            })
        
        logger.info(f"Found {len(images)} images")
        
        return json.dumps({
            "status": "success",
            "count": len(images),
            "images": images,
            "bucket": S3_IMAGES_BUCKET,
            "prefix": IMAGES_PREFIX
        })
    
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        return json.dumps({
            "error": str(e),
            "status": "failed"
        })


# =============================================================================
# DELETE IMAGE
# =============================================================================
@tool
def delete_image(image_key: str) -> str:
    """
    Delete a generated image from S3.
    
    Args:
        image_key: S3 key of the image to delete (e.g., "user/project/images/image.png")
    
    Returns:
        JSON string with deletion status
    
    Example:
        delete_image("202/202/images/generated_20250124_123456_12345.png")
    """
    try:
        logger.info(f"Deleting image: s3://{S3_IMAGES_BUCKET}/{image_key}")
        
        s3_client.delete_object(
            Bucket=S3_IMAGES_BUCKET,
            Key=image_key
        )
        
        logger.info(f"✓ Image deleted")
        
        return json.dumps({
            "status": "success",
            "message": f"Image deleted: {image_key}",
            "s3_uri": f"s3://{S3_IMAGES_BUCKET}/{image_key}"
        })
    
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        return json.dumps({
            "error": str(e),
            "status": "failed"
        })


# =============================================================================
# IMAGE-TO-IMAGE GENERATION (Using existing image as reference)
# =============================================================================
@tool
def generate_image_from_image(
    source_image: str,
    prompt: str,
    strength: float = 0.75,
    model: str = 'sd3-large',
    seed: Optional[int] = None
) -> str:
    """
    Generate a new image based on an existing image (image-to-image generation).
    
    Args:
        source_image: Source image (S3 URI, S3 key, or local path)
        prompt: Text description of how to modify the image
        strength: How much to change the image (0.0 = no change, 1.0 = completely new)
        model: Model to use (default: 'sd3-large')
        seed: Random seed for reproducibility (optional)
    
    Returns:
        JSON string containing generated image details and S3 location
    
    Example:
        generate_image_from_image(
            "s3://bucket/original.png",
            "Turn this into a watercolor painting",
            strength=0.6
        )
    """
    try:
        # Validate model supports image-to-image
        if model not in ['sd3-large', 'stable-image-core', 'sdxl']:
            return json.dumps({
                "error": f"Model '{model}' doesn't support image-to-image. Use 'sd3-large', 'stable-image-core', or 'sdxl'",
                "status": "failed"
            })
        
        # Load source image
        image_bytes = None
        if source_image.startswith('s3://'):
            parts = source_image.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_bytes = response['Body'].read()
        elif '/' in source_image:
            response = s3_client.get_object(Bucket=S3_IMAGES_BUCKET, Key=source_image)
            image_bytes = response['Body'].read()
        else:
            with open(source_image, 'rb') as f:
                image_bytes = f.read()
        
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 4294967295)
        
        model_id = AVAILABLE_IMAGE_MODELS[model]
        
        # Prepare request
        if model == 'sdxl':
            request_body = {
                "text_prompts": [{"text": prompt}],
                "init_image": base64_image,
                "cfg_scale": 10,
                "image_strength": strength,
                "seed": seed,
                "steps": 30
            }
        else:
            request_body = {
                "prompt": prompt,
                "image": base64_image,
                "mode": "image-to-image",
                "strength": strength,
                "seed": seed
            }
        
        logger.info(f"Generating image-to-image with {model_id}")
        
        # Invoke model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        
        # Extract generated image
        if model == 'sdxl':
            base64_output = response_body['artifacts'][0]['base64']
        else:
            base64_output = response_body['images'][0]
        
        output_bytes = base64.b64decode(base64_output)
        
        # Save to S3
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        image_filename = f"img2img_{timestamp}_{seed}.png"
        image_key = f"{IMAGES_PREFIX}{image_filename}"
        
        s3_client.put_object(
            Bucket=S3_IMAGES_BUCKET,
            Key=image_key,
            Body=output_bytes,
            ContentType='image/png',
            Metadata={
                'prompt': prompt[:1024],
                'model': model,
                'seed': str(seed),
                'strength': str(strength),
                'type': 'image-to-image',
                'generated_at': timestamp
            }
        )
        
        s3_uri = f"s3://{S3_IMAGES_BUCKET}/{image_key}"
        s3_url = f"https://{S3_IMAGES_BUCKET}.s3.amazonaws.com/{image_key}"
        
        logger.info(f"✓ Image-to-image generated: {s3_uri}")
        
        return json.dumps({
            "status": "success",
            "s3_uri": s3_uri,
            "s3_url": s3_url,
            "image_key": image_key,
            "metadata": {
                "prompt": prompt,
                "model": model,
                "seed": seed,
                "strength": strength,
                "type": "image-to-image"
            }
        })
    
    except Exception as e:
        logger.error(f"Error in image-to-image generation: {str(e)}")
        return json.dumps({
            "error": str(e),
            "status": "failed"
        })


# =============================================================================
# TOOL REGISTRY FOR EASY ACCESS
# =============================================================================
IMAGE_TOOLS = {
    'generate_image': generate_image,
    'analyze_image': analyze_image,
    'list_generated_images': list_generated_images,
    'delete_image': delete_image,
    'generate_image_from_image': generate_image_from_image
}


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
"""
USAGE EXAMPLES:
===============

1. Generate an image:
   result = generate_image("A futuristic cityscape at sunset")
   # Returns S3 URI and URL

2. Generate with specific model:
   result = generate_image(
       "A portrait of a cat wearing a hat",
       model="stable-image-ultra",
       negative_prompt="blurry, low quality"
   )

3. Analyze an image:
   result = analyze_image(
       "s3://bucket/image.png",
       "What objects are visible in this image?"
   )

4. List all generated images:
   result = list_generated_images(limit=50)

5. Image-to-image transformation:
   result = generate_image_from_image(
       "s3://bucket/original.png",
       "Convert to oil painting style",
       strength=0.7
   )

6. Delete an image:
   result = delete_image("202/202/images/generated_20250124_123456.png")


INTEGRATION WITH AGENT SYSTEM:
===============================

# Add to tool_map in orchestrator
tool_map = {
    'http_request': http_request,
    'search_documents': search_documents,
    'generate_image': generate_image,
    'analyze_image': analyze_image,
    'list_generated_images': list_generated_images,
    'delete_image': delete_image,
    'generate_image_from_image': generate_image_from_image
}


ENVIRONMENT VARIABLES NEEDED:
==============================

S3_IMAGES_BUCKET=qubitz-customer-prod-v2
USER_ID=202
PROJECT_ID=202
BEDROCK_REGION=us-west-2
VISION_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0


REQUIRED IAM PERMISSIONS:
=========================

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/stability.*",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::qubitz-customer-prod-v2/*",
        "arn:aws:s3:::qubitz-customer-prod-v2"
      ]
    }
  ]
}


ENABLE MODELS IN BEDROCK:
=========================

1. Go to AWS Console → Amazon Bedrock → Model access
2. Enable the following models:
   - Stability AI Stable Diffusion 3 Large
   - Stability AI Stable Image Ultra
   - Stability AI Stable Image Core
   - Anthropic Claude 3.5 Sonnet v2

Note: Available regions for these models:
- Stability AI models: us-west-2, us-east-1
- Claude 3.5 Sonnet: Multiple regions including us-east-1, us-west-2
"""