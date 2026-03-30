#!/usr/bin/env python3
"""
COCO Karpathy Dataset Loader - Quick Reference & Usage Examples

Run this script to test the dataset loader with mock data.
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, '/workspaces/lavis-ai')

from load_coco_karpathy import COCOKarpathyLoader, load_coco_karpathy, load_coco_karpathy_with_ids


def create_mock_karpathy_json():
    """Create a mock Karpathy JSON file for testing."""
    
    mock_data = {
        "images": [
            {
                "sentenceids": [0, 1, 2, 3, 4],
                "imgid": 391895,
                "sentences": [
                    {"raw": "A man with a red helmet on a small moped on a dirt road."},
                    {"raw": "Man riding a motor bike on a dirt road on the countryside."},
                    {"raw": "A man in a white shirt and helmet sitting on a motorcycle."},
                    {"raw": "A motorcycle on a dirt ground."},
                    {"raw": "A man riding a motorcycle on a dirt road."},
                ],
                "split": "train",
                "filename": "train2014/COCO_train2014_000000391895.jpg"
            },
            {
                "sentenceids": [5, 6, 7, 8, 9],
                "imgid": 522418,
                "sentences": [
                    {"raw": "A woman in a white shirt and shorts is kicking a soccer ball."},
                    {"raw": "A woman kicking a soccer ball in a field."},
                    {"raw": "A girl in white clothing is kicking a black and white soccer ball."},
                    {"raw": "A woman playing soccer on a grass field."},
                    {"raw": "The woman is kicking the soccer ball."},
                ],
                "split": "train",
                "filename": "train2014/COCO_train2014_000000522418.jpg"
            },
            {
                "sentenceids": [10, 11, 12, 13, 14],
                "imgid": 184613,
                "sentences": [
                    {"raw": "A laptop computer sitting on top of a desk."},
                    {"raw": "There is a laptop on the wooden table."},
                    {"raw": "A laptop is open on a wooden desk."},
                    {"raw": "A silver laptop sits on a brown wooden desk."},
                    {"raw": "A computer laptop is on the wooden table."},
                ],
                "split": "val",
                "filename": "val2014/COCO_val2014_000000184613.jpg"
            },
            {
                "sentenceids": [15, 16, 17, 18, 19],
                "imgid": 268024,
                "sentences": [
                    {"raw": "A man and woman standing next to each other."},
                    {"raw": "A couple standing together outside."},
                    {"raw": "Two people posing for a photograph."},
                    {"raw": "A man and woman smiling at the camera."},
                    {"raw": "Two people standing side by side."},
                ],
                "split": "val",
                "filename": "val2014/COCO_val2014_000000268024.jpg"
            },
        ]
    }
    
    return mock_data


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_subsection(title):
    """Print a subsection header."""
    print(f"\n{title}")
    print(f"{'-'*70}")


def demo_1_basic_loading():
    """Demo 1: Basic dataset loading."""
    print_section("DEMO 1: Basic Dataset Loading")
    
    # Create mock data
    mock_data = create_mock_karpathy_json()
    json_path = "/tmp/coco_karpathy_demo.json"
    with open(json_path, "w") as f:
        json.dump(mock_data, f, indent=2)
    
    print_subsection("Loading training split")
    
    loader = COCOKarpathyLoader(json_path, split="train", verbose=True)
    
    print(f"\n✓ Successfully loaded {len(loader)} training samples")


def demo_2_data_access():
    """Demo 2: Accessing data."""
    print_section("DEMO 2: Accessing Dataset Items")
    
    mock_data = create_mock_karpathy_json()
    json_path = "/tmp/coco_karpathy_demo.json"
    with open(json_path, "w") as f:
        json.dump(mock_data, f, indent=2)
    
    loader = COCOKarpathyLoader(json_path, split="train", verbose=False)
    
    print_subsection("First item in training set:")
    item = loader[0]
    print(f"Image path: {item['image_path']}")
    print(f"Image ID:   {item['image_id']}")
    print(f"Caption:    {item['caption']}")
    
    print_subsection("All training samples:")
    for i, item in enumerate(loader.get_all(), 1):
        print(f"{i}. {item['image_path']}")
        print(f"   {item['caption']}")


def demo_3_multiple_splits():
    """Demo 3: Loading multiple splits."""
    print_section("DEMO 3: Multiple Dataset Splits")
    
    mock_data = create_mock_karpathy_json()
    json_path = "/tmp/coco_karpathy_demo.json"
    with open(json_path, "w") as f:
        json.dump(mock_data, f, indent=2)
    
    splits = ["train", "val"]
    
    print_subsection("Loading all splits:")
    print(f"{'Split':<10} {'Samples':<15} {'Images':<15}")
    print("-" * 40)
    
    for split in splits:
        loader = COCOKarpathyLoader(json_path, split=split, verbose=False)
        
        # Count unique images
        unique_images = set(item['image_path'] for item in loader.get_all())
        
        print(f"{split:<10} {len(loader):<15} {len(unique_images):<15}")


def demo_4_caption_cleaning():
    """Demo 4: Caption cleaning process."""
    print_section("DEMO 4: Caption Processing & Cleaning")
    
    print_subsection("Steps in caption cleaning:")
    print("1. Load: coco_data['images']")
    print("2. Filter: if image['split'] == split")
    print("3. Get: image_path = image['filename']")
    print("4. Extract: caption = sentence['raw']")
    print("5. Clean: caption.lower().strip()")
    print("6. Store: (image_path, caption)")
    
    print_subsection("Before & After Examples:")
    
    raw_captions = [
        "A man with a red helmet on a small moped on a dirt road.",
        "  A woman kicking a soccer ball in a field.  ",
        "THERE IS A LAPTOP ON THE WOODEN TABLE.",
    ]
    
    for i, raw in enumerate(raw_captions, 1):
        cleaned = raw.lower().strip()
        print(f"\n{i}. Raw:     '{raw}'")
        print(f"   Cleaned: '{cleaned}'")


def demo_5_validation():
    """Demo 5: Dataset validation."""
    print_section("DEMO 5: Dataset Validation")
    
    mock_data = create_mock_karpathy_json()
    json_path = "/tmp/coco_karpathy_demo.json"
    with open(json_path, "w") as f:
        json.dump(mock_data, f, indent=2)
    
    loader = COCOKarpathyLoader(json_path, split="train", verbose=False)
    
    print_subsection("Dataset Statistics:")
    print(f"Total samples:  {len(loader)}")
    print(f"Total images:   {len(set(item['image_path'] for item in loader.get_all()))}")
    print(f"Avg captions/img: {len(loader) / len(set(item['image_path'] for item in loader.get_all())):.2f}")
    
    print_subsection("Sample Captions (first 2):")
    for i, item in enumerate(loader.get_samples(2), 1):
        print(f"{i}. {item['caption']}")
    
    print()


def demo_6_function_usage():
    """Demo 6: Using convenience functions."""
    print_section("DEMO 6: Using Convenience Functions")
    
    mock_data = create_mock_karpathy_json()
    json_path = "/tmp/coco_karpathy_demo.json"
    with open(json_path, "w") as f:
        json.dump(mock_data, f, indent=2)
    
    print_subsection("Function 1: load_coco_karpathy()")
    dataset = load_coco_karpathy(json_path, split="train", verbose=False)
    print(f"Returns: List of (image_path, caption) tuples")
    print(f"Length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
    
    print_subsection("Function 2: load_coco_karpathy_with_ids()")
    dataset_with_ids = load_coco_karpathy_with_ids(json_path, split="train", verbose=False)
    print(f"Returns: List of dicts with keys: image_path, caption, image_id")
    print(f"Length: {len(dataset_with_ids)}")
    print(f"First item: {dataset_with_ids[0]}")


def main():
    """Run all demonstrations."""
    
    print("\n" + "="*70)
    print("  COCO Karpathy JSON Dataset Loader - Usage Demonstrations")
    print("="*70)
    
    try:
        demo_1_basic_loading()
        demo_2_data_access()
        demo_3_multiple_splits()
        demo_4_caption_cleaning()
        demo_5_validation()
        demo_6_function_usage()
        
        print_section("SUMMARY")
        print("""
✓ All demonstrations completed successfully!

Key Features:
  • Load COCO data from Karpathy JSON format
  • Filter by split (train/val/test)
  • Extract and clean captions
  • Track image IDs
  • Print validation info (dataset size, sample captions)

Quick Start:
  from load_coco_karpathy import load_coco_karpathy
  
  dataset = load_coco_karpathy('annotations.json', split='train')
  image_path, caption = dataset[0]
  print(f"Dataset size: {len(dataset)}")

See COCO_KARPATHY_README.md for detailed documentation.
        """)
        
    except Exception as e:
        print(f"\n❌ Error during demonstrations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
