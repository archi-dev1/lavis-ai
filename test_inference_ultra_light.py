#!/usr/bin/env python3
"""
Ultra-light BLIP-2 inference test (NO training)
- Load model (current optimized state)
- Generate captions for 2-3 sample images
- Validate captions are generated
"""

import sys
sys.path.insert(0, '/workspaces/lavis-ai')

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io

def create_dummy_image(width=224, height=224, name="sample"):
    """Create a dummy RGB image for testing."""
    # Create a colorful gradient image
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient patterns
    for i in range(height):
        for j in range(width):
            arr[i, j, 0] = int(255 * (i / height))  # Red gradient
            arr[i, j, 1] = int(255 * (j / width))   # Green gradient
            arr[i, j, 2] = int(255 * (1 - i/height))  # Blue gradient
    
    img = Image.fromarray(arr, 'RGB')
    return img, name


def load_blip2_model():
    """
    Load BLIP-2 model with optimized configuration:
    - Vision encoder: PVT v2 b2 (49 patches)
    - Query tokens: 16
    - LoRA: 1.04% trainable
    """
    print("Loading BLIP-2 model (optimized configuration)...")
    
    try:
        from lavis.models import load_model_and_preprocess
        
        # Load model (uses optimized defaults from blip2_t5.py)
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain",
            is_eval=True,  # Evaluation mode (no gradients)
            device="cpu"   # Use CPU for ultra-light test
        )
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: cpu")
        print(f"  Vision encoder: PVT v2 b2 (optimized)")
        print(f"  Query tokens: 16 (optimized)")
        
        return model, vis_processors, txt_processors
    
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        print(f"  Error details: {type(e).__name__}")
        raise


def create_test_images():
    """Create 3 dummy test images."""
    images = []
    
    print("\nCreating test images...")
    
    # Image 1: Red-Green gradient
    arr1 = np.zeros((224, 224, 3), dtype=np.uint8)
    arr1[:, :, 0] = np.linspace(0, 255, 224)
    arr1[:, :, 1] = np.linspace(0, 255, 224).reshape(-1, 1)
    img1 = Image.fromarray(arr1, 'RGB')
    images.append(("image_1_red_green", img1))
    
    # Image 2: Blue-Yellow gradient
    arr2 = np.zeros((224, 224, 3), dtype=np.uint8)
    arr2[:, :, 0] = np.linspace(255, 0, 224).reshape(-1, 1)
    arr2[:, :, 1] = np.linspace(255, 0, 224).reshape(-1, 1)
    arr2[:, :, 2] = np.linspace(0, 255, 224)
    img2 = Image.fromarray(arr2, 'RGB')
    images.append(("image_2_blue_yellow", img2))
    
    # Image 3: Mixed color pattern
    arr3 = np.zeros((224, 224, 3), dtype=np.uint8)
    arr3[0:112, 0:112] = [255, 0, 0]      # Red quarter
    arr3[0:112, 112:224] = [0, 255, 0]    # Green quarter
    arr3[112:224, 0:112] = [0, 0, 255]    # Blue quarter
    arr3[112:224, 112:224] = [255, 255, 0]  # Yellow quarter
    img3 = Image.fromarray(arr3, 'RGB')
    images.append(("image_3_color_blocks", img3))
    
    print(f"✓ Created {len(images)} test images")
    return images


def generate_captions(model, vis_processors, images, num_beams=3, max_length=20):
    """Generate captions for images using optimized BLIP-2."""
    
    print(f"\nGenerating captions (num_beams={num_beams}, max_length={max_length})...")
    print("-" * 80)
    
    results = []
    
    with torch.no_grad():  # No gradients needed for inference
        for image_name, image in images:
            try:
                # Preprocess image
                processed_image = vis_processors["eval"](image).unsqueeze(0)
                
                # Generate caption
                caption = model.generate(
                    {"image": processed_image},
                    use_nucleus_sampling=False,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=5,
                )
                
                # Caption is a list, get first element
                caption_text = caption[0] if isinstance(caption, list) else caption
                
                results.append({
                    'image_name': image_name,
                    'caption': caption_text,
                    'status': '✓'
                })
                
                print(f"Image: {image_name}")
                print(f"Caption: {caption_text}")
                print()
                
            except Exception as e:
                print(f"✗ Error generating caption for {image_name}: {str(e)}")
                results.append({
                    'image_name': image_name,
                    'caption': f"ERROR: {str(e)}",
                    'status': '✗'
                })
    
    return results


def run_inference_test():
    """Run complete ultra-light inference test."""
    
    print("=" * 80)
    print("BLIP-2 ULTRA-LIGHT INFERENCE TEST (NO TRAINING)")
    print("=" * 80)
    
    try:
        # Step 1: Load model
        print("\n[Step 1] Loading Model")
        print("-" * 80)
        model, vis_processors, txt_processors = load_blip2_model()
        
        # Step 2: Create test images
        print("\n[Step 2] Creating Test Images")
        print("-" * 80)
        images = create_test_images()
        
        # Step 3: Generate captions
        print("\n[Step 3] Generating Captions")
        print("-" * 80)
        results = generate_captions(
            model, 
            vis_processors, 
            images,
            num_beams=3,
            max_length=20
        )
        
        # Step 4: Summary
        print("=" * 80)
        print("INFERENCE TEST SUMMARY")
        print("=" * 80)
        
        total = len(results)
        successful = sum(1 for r in results if r['status'] == '✓')
        failed = sum(1 for r in results if r['status'] == '✗')
        
        print(f"\nTotal images processed: {total}")
        print(f"Successful captions: {successful} ✓")
        print(f"Failed captions: {failed} ✗")
        
        print(f"\nGenerated Captions:")
        print("-" * 80)
        for result in results:
            status = "✓" if result['status'] == '✓' else "✗"
            print(f"{status} {result['image_name']}")
            print(f"   Caption: {result['caption']}")
            print()
        
        print("=" * 80)
        if failed == 0:
            print("✓ ALL TESTS PASSED - INFERENCE WORKING")
        else:
            print(f"⚠ {failed} TEST(S) FAILED")
        print("=" * 80)
        
        return failed == 0
    
    except Exception as e:
        print(f"\n✗ INFERENCE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_inference_test()
    exit(0 if success else 1)
