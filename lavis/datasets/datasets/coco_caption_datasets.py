"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random
import logging

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionInstructDataset, CaptionEvalDataset
from lavis.datasets.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

COCOCapDataset = CaptionDataset
COCOCapInstructDataset = CaptionInstructDataset


class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        """
        Get item with automatic retry on failure.
        Never returns None.
        """
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                ann = self.annotation[index]

                image_path = os.path.join(self.vis_root, ann["image"])
                image = Image.open(image_path).convert("RGB")

                image = self.vis_processor(image)

                img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

                return {
                    "image": image,
                    "image_id": img_id,
                    "instance_id": ann["instance_id"],
                }
            except Exception as e:
                retry_count += 1
                
                if retry_count < max_retries:
                    index = random.randint(0, len(self.annotation) - 1)
                else:
                    logger.error(
                        f"Failed to load COCO eval sample after {max_retries} retries. "
                        f"Last error: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Cannot load valid COCO eval sample after {max_retries} retries"
                    ) from e
        
        raise RuntimeError("Unexpected error in COCOCapEvalDataset __getitem__")



class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        """
        Get item with automatic retry on failure.
        Never returns None.
        """
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                ann = self.annotation[index]

                image_path = os.path.join(self.vis_root, ann["image"])
                image = Image.open(image_path).convert("RGB")

                image = self.vis_processor(image)

                img_id = ann["img_id"]

                return {
                    "image": image,
                    "image_id": img_id,
                    "instance_id": ann["instance_id"],
                }
            except Exception as e:
                retry_count += 1
                
                if retry_count < max_retries:
                    index = random.randint(0, len(self.annotation) - 1)
                else:
                    logger.error(
                        f"Failed to load NoCaps eval sample after {max_retries} retries. "
                        f"Last error: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Cannot load valid NoCaps eval sample after {max_retries} retries"
                    ) from e
        
        raise RuntimeError("Unexpected error in NoCapsEvalDataset __getitem__")


class COCOKarpathyDataset(BaseDataset):
    """
    COCO Dataset using Karpathy JSON format.
    
    Expected JSON format:
    {
        "images": [
            {
                "sentenceids": [...],
                "imgid": int,
                "sentences": [{"raw": "caption text", ...}],
                "split": "train|val|test",
                "filename": "image_filename.jpg"
            },
            ...
        ]
    }
    """
    
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, split="train"):
        """
        Args:
            vis_processor: Visual processor for images
            text_processor: Text processor for captions
            vis_root (string): Root directory of images
            ann_paths (list): Paths to Karpathy JSON annotation files
            split (string): Dataset split ('train', 'val', or 'test')
        """
        self.vis_root = vis_root
        self.split = split
        self.annotation = []
        
        # Load Karpathy JSON and extract image-caption pairs
        for ann_path in ann_paths:
            with open(ann_path, "r") as f:
                coco_data = json.load(f)
            
            # Process each image in the dataset
            for image in coco_data["images"]:
                # Step 2: Filter by split
                if image["split"] != self.split:
                    continue
                
                # Step 3: Get image path
                image_path = image["filename"]
                
                # Step 4 & 5: For each sentence, get and clean caption
                for sentence in image["sentences"]:
                    # Get raw caption
                    caption = sentence["raw"]
                    
                    # Step 6: Clean caption - lowercase and strip whitespace
                    caption = caption.lower().strip()
                    
                    # Store the annotation
                    self.annotation.append({
                        "image": image_path,
                        "caption": caption,
                        "image_id": image.get("imgid", image.get("cocoid", 0)),
                    })
        
        # Step 1: Print dataset size and sample captions
        print(f"\n[Dataset Info] COCO Karpathy {self.split.upper()}")
        print(f"Dataset size: {len(self.annotation)} image-caption pairs")
        
        if len(self.annotation) > 0:
            print(f"Sample captions:")
            for i, ann in enumerate(self.annotation[:2]):
                print(f"  {i+1}. {ann['caption']}")
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self._add_instance_ids()
    
    def __getitem__(self, index):
        """
        Get item with automatic retry on failure.
        
        Retries up to 10 times if:
        - Image loading fails
        - Image transform fails
        - Caption processing fails
        
        Never returns None.
        """
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                ann = self.annotation[index]
                
                image_path = os.path.join(self.vis_root, ann["image"])
                
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Process image
                image = self.vis_processor(image)
                
                # Process caption
                caption = self.text_processor(ann["caption"])
                
                # Validate caption
                if not caption or (isinstance(caption, str) and not caption.strip()):
                    raise ValueError(f"Empty caption for {image_path}")
                
                return {
                    "image": image,
                    "text_input": caption,
                    "image_id": ann["image_id"],
                }
            except Exception as e:
                retry_count += 1
                
                if retry_count < max_retries:
                    # Retry with random index
                    index = random.randint(0, len(self.annotation) - 1)
                else:
                    # Max retries exceeded
                    logger.error(
                        f"Failed to load Karpathy sample after {max_retries} retries. "
                        f"Last error: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Cannot load valid Karpathy sample after {max_retries} retries"
                    ) from e
        
        raise RuntimeError("Unexpected error in COCOKarpathyDataset __getitem__")


class COCOKarpathyEvalDataset(BaseDataset):
    """
    COCO Evaluation Dataset using Karpathy JSON format.
    For evaluation (no caption processing needed).
    """
    
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, split="val"):
        """
        Args:
            vis_processor: Visual processor for images
            text_processor: Text processor (not used for eval)
            vis_root (string): Root directory of images
            ann_paths (list): Paths to Karpathy JSON annotation files
            split (string): Dataset split ('val' or 'test')
        """
        self.vis_root = vis_root
        self.split = split
        self.annotation = []
        
        # Load Karpathy JSON and extract images
        for ann_path in ann_paths:
            with open(ann_path, "r") as f:
                coco_data = json.load(f)
            
            for image in coco_data["images"]:
                # Filter by split
                if image["split"] != self.split:
                    continue
                
                image_path = image["filename"]
                
                # For evaluation, store all captions for each image
                captions = [sentence["raw"].lower().strip() for sentence in image["sentences"]]
                
                self.annotation.append({
                    "image": image_path,
                    "captions": captions,
                    "image_id": image.get("imgid", image.get("cocoid", 0)),
                })
        
        print(f"\n[Dataset Info] COCO Karpathy Eval {self.split.upper()}")
        print(f"Dataset size: {len(self.annotation)} images")
        
        if len(self.annotation) > 0:
            print(f"Sample captions:")
            for i, ann in enumerate(self.annotation[:2]):
                print(f"  Image {i+1}: {ann['captions'][0]}")
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self._add_instance_ids()
    
    def __getitem__(self, index):
        """
        Get item with automatic retry on failure.
        
        Retries up to 10 times if:
        - Image loading fails
        - Image transform fails
        
        Never returns None.
        """
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                ann = self.annotation[index]
                
                image_path = os.path.join(self.vis_root, ann["image"])
                
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Process image
                image = self.vis_processor(image)
                
                return {
                    "image": image,
                    "captions": ann["captions"],
                    "image_id": ann["image_id"],
                    "instance_id": ann["instance_id"],
                }
            except Exception as e:
                retry_count += 1
                
                if retry_count < max_retries:
                    # Retry with random index
                    index = random.randint(0, len(self.annotation) - 1)
                else:
                    # Max retries exceeded
                    logger.error(
                        f"Failed to load Karpathy eval sample after {max_retries} retries. "
                        f"Last error: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Cannot load valid Karpathy eval sample after {max_retries} retries"
                    ) from e
        
        raise RuntimeError("Unexpected error in COCOKarpathyEvalDataset __getitem__")