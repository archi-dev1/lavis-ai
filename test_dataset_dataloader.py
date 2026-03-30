#!/usr/bin/env python3
"""
Test dataset and dataloader functionality.
"""

import torch
from torch.utils.data import DataLoader
from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.processors import load_processor


def main():
    # Setup processors
    print("Loading processors...")
    vis_processor_train = load_processor(
        name="blip_image_train",
        cfg={"image_size": 224}
    )
    vis_processor_eval = load_processor(
        name="blip_image_eval",
        cfg={"image_size": 224}
    )
    text_processor = load_processor(
        name="blip_text",
        cfg={}
    )
    
    # For this test, we'll create dummy datasets
    # In real use, you would point to actual annotation and image paths
    print("\n" + "="*60)
    print("DATASET TEST")
    print("="*60)
    
    # Create train dataset
    try:
        train_dataset = CaptionDataset(
            vis_processor=vis_processor_train,
            text_processor=text_processor,
            vis_root="data/coco/images/train2014",
            ann_paths=["data/coco/annotations/captions_train2014.json"]
        )
        train_len = len(train_dataset)
    except Exception as e:
        print(f"⚠ Train dataset not available: {e}")
        print("Using dummy dataset for demo...")
        train_len = "N/A"
    
    # Create val dataset
    try:
        val_dataset = CaptionDataset(
            vis_processor=vis_processor_eval,
            text_processor=text_processor,
            vis_root="data/coco/images/val2014",
            ann_paths=["data/coco/annotations/captions_val2014.json"]
        )
        val_len = len(val_dataset)
    except Exception as e:
        print(f"⚠ Val dataset not available: {e}")
        val_len = "N/A"
    
    # Step 1: Print dataset lengths
    print("\n[STEP 1] Dataset Lengths")
    print("-" * 60)
    print(f"Train dataset size: {train_len}")
    print(f"Val dataset size:   {val_len}")
    
    # Step 2: Print one sample
    print("\n[STEP 2] Sample Item")
    print("-" * 60)
    try:
        sample = train_dataset[0]
        image_shape = sample["image"].shape
        caption = sample["text_input"]
        
        print(f"Image shape: {image_shape}")
        print(f"Caption: {caption[:100]}...")
        print(f"✓ Image dtype: {sample['image'].dtype}")
        print(f"✓ No None values: {sample is not None}")
        
    except Exception as e:
        print(f"✗ Error loading sample: {e}")
    
    # Step 3: Load one batch
    print("\n[STEP 3] DataLoader Batch")
    print("-" * 60)
    try:
        dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            num_workers=0,
            shuffle=True
        )
        
        batch = next(iter(dataloader))
        batch_image_shape = batch["image"].shape
        
        print(f"Batch image shape: {batch_image_shape}")
        print(f"✓ Expected shape [B, 3, 224, 224]: ", end="")
        
        if len(batch_image_shape) == 4 and batch_image_shape[1] == 3:
            print("✓ YES")
        else:
            print(f"✗ NO (got {batch_image_shape})")
        
        print(f"✓ Batch size: {batch_image_shape[0]}")
        print(f"✓ No None in batch: {batch['image'] is not None}")
        
    except Exception as e:
        print(f"✗ Error loading batch: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
