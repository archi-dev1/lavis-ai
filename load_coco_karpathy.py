#!/usr/bin/env python3
"""
COCO Karpathy JSON Dataset Loader

This script demonstrates loading COCO data from Karpathy JSON format.

Karpathy JSON Format:
{
    "images": [
        {
            "sentenceids": [...],
            "imgid": <int>,
            "sentences": [
                {"raw": "caption text", ...},
                ...
            ],
            "split": "train|val|test",
            "filename": "image_filename.jpg"
        },
        ...
    ]
}

Steps:
1. Load: coco_data["images"]
2. For each image: if image["split"] == split
3. Get: image_path = image["filename"]
4. For each sentence: caption = sentence["raw"]
5. Store: (image_path, caption)
6. Clean: caption = caption.lower().strip()
"""

import os
import json
from typing import List, Dict, Tuple, Optional


class COCOKarpathyLoader:
    """COCO Karpathy JSON format dataset loader."""
    
    def __init__(self, json_path: str, split: str = "train", verbose: bool = True):
        """
        Initialize the COCO Karpathy loader.
        
        Args:
            json_path (str): Path to Karpathy JSON file
            split (str): Dataset split ('train', 'val', or 'test')
            verbose (bool): Print dataset info
        """
        self.json_path = json_path
        self.split = split
        self.verbose = verbose
        self.data = []
        self._load_and_process()
    
    def _load_and_process(self) -> None:
        """
        Load and process COCO data from Karpathy JSON.
        
        Steps:
        1. Load: coco_data["images"]
        2. For each image: if image["split"] == self.split
        3. Get: image_path = image["filename"]
        4. For each sentence: caption = sentence["raw"]
        5. Store: (image_path, caption)
        6. Clean: caption = caption.lower().strip()
        """
        
        # Step 1: Load JSON
        with open(self.json_path, "r") as f:
            coco_data = json.load(f)
        
        image_count = 0
        caption_count = 0
        
        # Iterate through images
        for image in coco_data["images"]:
            # Step 2: Filter by split
            if image["split"] != self.split:
                continue
            
            image_count += 1
            
            # Step 3: Get image path
            image_path = image["filename"]
            
            # Get image ID
            image_id = image.get("imgid", image.get("cocoid", 0))
            
            # Step 4: Process each sentence
            for sentence in image["sentences"]:
                # Get raw caption
                caption = sentence["raw"]
                
                # Step 5 & 6: Clean caption - lowercase and strip whitespace
                caption = caption.lower().strip()
                
                # Store the image-caption pair
                self.data.append({
                    "image_path": image_path,
                    "caption": caption,
                    "image_id": image_id,
                })
                
                caption_count += 1
        
        # Validation: Print dataset size and sample captions
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"COCO Karpathy Dataset - {self.split.upper()}")
            print(f"{'='*60}")
            print(f"Images processed: {image_count}")
            print(f"Total image-caption pairs: {caption_count}")
            print(f"Average captions per image: {caption_count / image_count:.2f}" if image_count > 0 else "")
            
            if len(self.data) > 0:
                print(f"\n[Sample Captions]")
                print(f"{'─'*60}")
                for i in range(min(2, len(self.data))):
                    item = self.data[i]
                    print(f"Sample {i+1}:")
                    print(f"  Path: {item['image_path']}")
                    print(f"  ID:   {item['image_id']}")
                    print(f"  Text: {item['caption']}")
                    print()
            
            print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Dict[str, str]:
        """Get item by index."""
        return self.data[index]
    
    def get_samples(self, num_samples: int = 5) -> List[Dict[str, str]]:
        """Get first N samples."""
        return self.data[:num_samples]
    
    def get_all(self) -> List[Dict[str, str]]:
        """Get all data."""
        return self.data


def load_coco_karpathy(
    json_path: str,
    split: str = "train",
    verbose: bool = True
) -> List[Tuple[str, str]]:
    """
    Load COCO dataset from Karpathy JSON format.
    
    Args:
        json_path (str): Path to Karpathy JSON file
        split (str): Dataset split ('train', 'val', or 'test')
        verbose (bool): Print dataset info
    
    Returns:
        List of (image_path, caption) tuples
    
    Example:
        >>> dataset = load_coco_karpathy('annotations_karpathy.json', split='train')
        >>> print(f"Dataset size: {len(dataset)}")
        >>> image_path, caption = dataset[0]
    """
    loader = COCOKarpathyLoader(json_path, split=split, verbose=verbose)
    return [(item["image_path"], item["caption"]) for item in loader.get_all()]


def load_coco_karpathy_with_ids(
    json_path: str,
    split: str = "train",
    verbose: bool = True
) -> List[Dict[str, str]]:
    """
    Load COCO dataset from Karpathy JSON with image IDs.
    
    Args:
        json_path (str): Path to Karpathy JSON file
        split (str): Dataset split ('train', 'val', or 'test')
        verbose (bool): Print dataset info
    
    Returns:
        List of dicts with keys: image_path, caption, image_id
    """
    loader = COCOKarpathyLoader(json_path, split=split, verbose=verbose)
    return loader.get_all()


if __name__ == "__main__":
    """Example usage."""
    
    # Example: Create a mock Karpathy JSON for testing
    mock_data = {
        "images": [
            {
                "sentenceids": [0, 1],
                "imgid": 1,
                "sentences": [
                    {"raw": "A  young boy is jumping in the air"},
                    {"raw": "A child is jumping on a bed"},
                ],
                "split": "train",
                "filename": "train2014/COCO_train2014_000000000001.jpg"
            },
            {
                "sentenceids": [2, 3],
                "imgid": 2,
                "sentences": [
                    {"raw": "A woman is standing in front of a mirror"},
                    {"raw": "A person is looking at their reflection"},
                ],
                "split": "train",
                "filename": "train2014/COCO_train2014_000000000002.jpg"
            },
            {
                "sentenceids": [4],
                "imgid": 3,
                "sentences": [
                    {"raw": "A group of people sitting around a table"},
                ],
                "split": "val",
                "filename": "val2014/COCO_val2014_000000000003.jpg"
            },
        ]
    }
    
    # Save mock data to file
    mock_json_path = "/tmp/coco_karpathy_mock.json"
    os.makedirs("/tmp", exist_ok=True)
    
    with open(mock_json_path, "w") as f:
        json.dump(mock_data, f, indent=2)
    
    print("Loading COCO Karpathy Dataset")
    print("="*60)
    
    # Example 1: Load training split
    print("\n[Example 1] Loading training split")
    print("-"*60)
    train_dataset = load_coco_karpathy(mock_json_path, split="train", verbose=True)
    print(f"Loaded {len(train_dataset)} samples")
    
    # Example 2: Load validation split
    print("\n[Example 2] Loading validation split")
    print("-"*60)
    val_dataset = load_coco_karpathy(mock_json_path, split="val", verbose=True)
    print(f"Loaded {len(val_dataset)} samples")
    
    # Example 3: Using the loader class directly
    print("\n[Example 3] Using COCOKarpathyLoader class")
    print("-"*60)
    loader = COCOKarpathyLoader(mock_json_path, split="train", verbose=False)
    print(f"Dataset size: {len(loader)}")
    
    # Get first sample
    first_sample = loader[0]
    print(f"First sample:")
    print(f"  Image: {first_sample['image_path']}")
    print(f"  Caption: {first_sample['caption']}")
    print(f"  ID: {first_sample['image_id']}")
    
    # Example 4: Get all data with IDs
    print("\n[Example 4] Get all data with IDs")
    print("-"*60)
    all_data = load_coco_karpathy_with_ids(mock_json_path, split="train", verbose=False)
    for i, item in enumerate(all_data[:2], 1):
        print(f"Item {i}: {item['image_path']} -> {item['caption']}")
    
    print("\n" + "="*60)
    print("✓ All examples completed successfully!")
    print("="*60 + "\n")
