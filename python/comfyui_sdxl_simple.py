#!/usr/bin/env python3
"""ComfyUI SDXL simple workflow - 1920x1080, 4 images"""
import requests
import time
import json
from comfyui_client import queue_prompt, wait_for_completion

COMFYUI_URL = "http://127.0.0.1:8192"

LATENT_W, LATENT_H = 1920, 1080
BATCH = 4
STEPS = 25
CFG = 7.0
SEED = 42

POSITIVE = (
    "japanese anime woman in traditional kimono, floral print pattern on silk fabric, "
    "holding paper umbrella, flowing sleeves, cherry blossom trees in background, "
    "soft moonlight, garden bridge, cherry petals falling, serene expression, "
    "detailed hair ornament, long black hair, elegant standing pose, morning mist, "
    "ultra detailed, sharp focus, beautiful lighting, cinematic composition"
)
NEGATIVE = (
    "breasts, naked, nsfw, deformed, blurry, low quality, watermark, text, "
    "signature, extra limbs, bad anatomy, bad hands, missing fingers, "
    "worst quality, low resolution, cropped"
)

# SDXL simple workflow
# 1: Load model (KJ) → 2: Empty latent (1920x1080, batch=4) → 3: CLIP + (pos) → 4: CLIP - (neg) → 5: KSampler → 6: VAEDecode → 7: Save
workflow = {
    "1": {
        "class_type": "CheckpointLoaderKJ",
        "inputs": {
            "ckpt_name": "JANKUTrainedChenkinNoobai_v69.safetensors",
            "weight_dtype": "fp8_e4m3fn",
            "compute_dtype": "default",
            "patch_cublaslinear": False,
            "sage_attention": "disabled",
            "enable_fp16_accumulation": False
        }
    },
    "2": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": LATENT_W,
            "height": LATENT_H,
            "batch_size": BATCH
        }
    },
    "3": {
        "class_type": "CLIPTextEncodeSDXL",
        "inputs": {
            "clip": ["1", 1],
            "width": LATENT_W,
            "height": LATENT_H,
            "crop_w": 0,
            "crop_h": 0,
            "target_width": LATENT_W,
            "target_height": LATENT_H,
            "text_g": POSITIVE,
            "text_l": POSITIVE
        }
    },
    "4": {
        "class_type": "CLIPTextEncodeSDXL",
        "inputs": {
            "clip": ["1", 1],
            "width": LATENT_W,
            "height": LATENT_H,
            "crop_w": 0,
            "crop_h": 0,
            "target_width": LATENT_W,
            "target_height": LATENT_H,
            "text_g": NEGATIVE,
            "text_l": NEGATIVE
        }
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "seed": SEED,
            "steps": STEPS,
            "cfg": CFG,
            "sampler_name": "euler_ancestral",
            "scheduler": "karras",
            "positive": ["3", 0],
            "negative": ["4", 0],
            "latent_image": ["2", 0],
            "denoise": 1.0
        }
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["5", 0],
            "vae": ["1", 2]
        }
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["6", 0],
            "filename_prefix": "sdxl_kimono_1920"
        }
    }
}

print(f"=== SDXL Workflow ===")
print(f"Model: JANKUTrainedChenkinNoobai_v69.safetensors")
print(f"Size: {LATENT_W}×{LATENT_H}, Batch: {BATCH}, Steps: {STEPS}, CFG: {CFG}")
print()

print("Submitting...")
prompt_id = queue_prompt(workflow)
print(f"Prompt ID: {prompt_id}")
print("Waiting for completion (this may take a few minutes)...")

result = wait_for_completion(prompt_id, timeout=900)
status = result.get("status", {})
print(f"\nStatus: {status.get('status', 'unknown')}")
if status.get("errors"):
    print(f"Errors: {status['errors']}")
print("Done! Check D:\share\output for images.")
