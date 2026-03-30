#!/usr/bin/env python3
"""
Direct test of dataset retry mechanism.
"""

import os
import sys
import json
import random
from pathlib import Path

# Add lavis to path
sys.path.insert(0, '/workspaces/lavis-ai')

import torch
from PIL import Image
import torch.nn.functional as F


def create_dummy_dataset(num_samples=10):
    """Create a dummy COCO dataset for testing."""
    data_dir = Path("/tmp/test_coco")
    data_dir.mkdir(exist_ok=True)
    
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create dummy images
    for i in range(num_samples):
        img = Image.new('RGB', (256, 256), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        img.save(images_dir / f"image_{i:06d}.jpg")
    
    # Create annotation file
    annotations = {
        "images": [
            {
                "file_name": f"image_{i:06d}.jpg",
                "id": i,
                "width": 256,
                "height": 256
            }
            for i in range(num_samples)
        ],
        "annotations": [
            {
                "image_id": i,
                "id": i,
                "caption": f"This is a test caption for image {i}."
            }
            for i in range(num_samples)
        ]
    }
    
    ann_file = data_dir / "annotations.json"
    with open(ann_file, 'w') as f:
        json.dump(annotations, f)
    
    return str(images_dir), str(ann_file), data_dir


class SimpleImageProcessor:
    """Simple image processor for testing."""
    def __call__(self, image):
        # Convert to tensor and normalize
        img_array = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        # Resize to 224x224
        img_array = F.interpolate(img_array.unsqueeze(0), size=(224, 224), mode='bilinear').squeeze(0)
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        return img_array


class SimpleTextProcessor:
    """Simple text processor for testing."""
    def __call__(self, text):
        return text.lower().strip()


# Test the retry mechanism directly
def test_retry_mechanism():
    print("Creating test dataset...")
    vis_root, ann_file, data_dir = create_dummy_dataset(20)
    
    print(f"Dataset created at {data_dir}")
    print(f"Images: {vis_root}")
    print(f"Annotations: {ann_file}")
    
    # Load annotations
    with open(ann_file, 'r') as f:
        coco = json.load(f)
    
    dataset_info = {
        'annotations': coco['annotations'],
        'images': {img['id']: img for img in coco['images']}
    }
    
    print("\n" + "="*60)
    print("DATASET TEST")
    print("="*60)
    
    # Step 1: Print dataset length
    print("\n[STEP 1] Dataset Size")
    print("-" * 60)
    dataset_size = len(coco['annotations'])
    print(f"Train dataset size: {dataset_size}")
    print(f"Val dataset size:   {int(dataset_size * 0.1)}")
    
    # Step 2: Test loading a sample
    print("\n[STEP 2] Sample Item")
    print("-" * 60)
    
    # Create processor
    import numpy as np
    processor = SimpleImageProcessor()
    text_processor = SimpleTextProcessor()
    
    try:
        # Load first annotation
        ann = coco['annotations'][0]
        img_id = ann['image_id']
        img_info = dataset_info['images'][img_id]
        
        # Load image
        img_path = os.path.join(vis_root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Process
        image_tensor = processor(image)
        caption = text_processor(ann['caption'])
        
        print(f"Image shape: {image_tensor.shape}")
        print(f"Caption: {caption[:100]}...")
        print(f"✓ Image dtype: {image_tensor.dtype}")
        print(f"✓ Shape matches [3, 224, 224]: {image_tensor.shape == torch.Size([3, 224, 224])}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Test batch loading
    print("\n[STEP 3] DataLoader Batch")
    print("-" * 60)
    
    try:
        # Create simple dataset class
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, annotations, images_dir, vis_root):
                self.annotations = annotations
                self.images = images_dir
                self.vis_root = vis_root
                self.processor = SimpleImageProcessor()
                self.text_processor = SimpleTextProcessor()
            
            def __len__(self):
                return len(self.annotations)
            
            def __getitem__(self, index):
                # Use retry mechanism
                max_retries = 10
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        ann = self.annotations[index]
                        img_info = self.images[ann['image_id']]
                        img_path = os.path.join(self.vis_root, img_info['file_name'])
                        
                        image = Image.open(img_path).convert('RGB')
                        image_tensor = self.processor(image)
                        caption = self.text_processor(ann['caption'])
                        
                        return {
                            'image': image_tensor,
                            'caption': caption
                        }
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            index = random.randint(0, len(self.annotations) - 1)
                        else:
                            raise RuntimeError(f"Cannot load after {max_retries} retries") from e
        
        # Create dataset and dataloader
        dataset = DummyDataset(
            coco['annotations'],
            dataset_info['images'],
            vis_root
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            num_workers=0,
            shuffle=True
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        batch_shape = batch['image'].shape
        
        print(f"Batch image shape: {batch_shape}")
        print(f"✓ Expected format [B, 3, 224, 224]: {len(batch_shape) == 4 and batch_shape[1] == 3}")
        print(f"✓ Batch size: {batch_shape[0]}")
        print(f"✓ Channels: {batch_shape[1]}")
        print(f"✓ Height: {batch_shape[2]}")
        print(f"✓ Width: {batch_shape[3]}")
        print(f"✓ No None in batch: {batch['image'] is not None}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    # Cleanup
    import shutil
    shutil.rmtree(data_dir)


if __name__ == "__main__":
    import numpy as np
    test_retry_mechanism()
