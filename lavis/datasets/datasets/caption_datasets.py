"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random
import logging
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image

logger = logging.getLogger(__name__)


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

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
                # Get annotation
                ann = self.annotation[index]
                
                # Load image
                image_path = os.path.join(self.vis_root, ann["image"])
                image = Image.open(image_path).convert("RGB")
                
                # Process image
                image = self.vis_processor(image)
                
                # Process caption
                caption = self.text_processor(ann["caption"])
                
                # Validate that caption is not empty
                if not caption or (isinstance(caption, str) and not caption.strip()):
                    raise ValueError(f"Empty caption for {image_path}")
                
                # Success - return data
                return {
                    "image": image,
                    "text_input": caption,
                    "image_id": ann["image_id"]
                }
                
            except Exception as e:
                retry_count += 1
                
                if retry_count < max_retries:
                    # Retry with random index
                    index = random.randint(0, len(self.annotation) - 1)
                else:
                    # Max retries exceeded - log error and raise
                    logger.error(
                        f"Failed to load sample after {max_retries} retries. "
                        f"Last error: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Cannot load valid sample after {max_retries} retries"
                    ) from e
        
        # Should never reach here
        raise RuntimeError("Unexpected error in __getitem__ retry loop")

class CaptionEvalDataset(BaseDataset, __DisplMixin):
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
        
        Retries up to 10 times if:
        - Image loading fails
        - Image transform fails
        
        Never returns None.
        """
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Get annotation
                ann = self.annotation[index]
                
                # Load image
                image_path = os.path.join(self.vis_root, ann["image"])
                image = Image.open(image_path).convert("RGB")
                
                # Process image
                image = self.vis_processor(image)
                
                # Success - return data
                return {
                    "image": image,
                    "image_id": ann["image_id"],
                    "instance_id": ann["instance_id"],
                }
                
            except Exception as e:
                retry_count += 1
                
                if retry_count < max_retries:
                    # Retry with random index
                    index = random.randint(0, len(self.annotation) - 1)
                else:
                    # Max retries exceeded - log error and raise
                    logger.error(
                        f"Failed to load eval sample after {max_retries} retries. "
                        f"Last error: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Cannot load valid eval sample after {max_retries} retries"
                    ) from e
        
        # Should never reach here
        raise RuntimeError("Unexpected error in eval __getitem__ retry loop")

class CaptionInstructDataset(CaptionDataset):
    def __getitem__(self, index):
        # Parent __getitem__ never returns None - always returns valid data or raises
        data = super().__getitem__(index)
        
        # data is guaranteed to be a dict, never None
        data['text_output'] = data["text_input"]
        data['text_input'] = self.text_processor("")
        
        return data