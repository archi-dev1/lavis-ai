# Complete __getitem__ Implementations

## File 1: lavis/datasets/datasets/caption_datasets.py

### Imports Added
```python
import random
import logging

logger = logging.getLogger(__name__)
```

---

## CaptionDataset.__getitem__()

**Purpose**: Load image-caption pairs from training set with automatic retry

```python
def __getitem__(self, index):
    """Load image-caption pair with automatic retry on errors."""
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            ann = self.annotation[index]
            
            # Load image
            image_path = os.path.join(self.vis_root, ann["image"])
            image = Image.open(image_path).convert("RGB")
            
            # Process image and caption
            image = self.vis_processor(image)
            caption = self.text_processor(ann["caption"])
            
            # Validate caption is not empty
            if not caption or (isinstance(caption, str) and not caption.strip()):
                raise ValueError(f"Empty caption after processing for {image_path}")
            
            return {
                "image": image,
                "text_input": caption,
            }
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                # Retry with random index
                index = random.randint(0, len(self.annotation) - 1)
            else:
                logger.error(
                    f"Failed to load CaptionDataset sample after {max_retries} retries. "
                    f"Index: {index}, Error: {str(e)}"
                )
                raise RuntimeError(
                    f"Cannot load valid caption sample after {max_retries} retries"
                ) from e
    
    # Should never reach here
    raise RuntimeError("Unexpected error in CaptionDataset.__getitem__ retry loop")
```

---

## CaptionEvalDataset.__getitem__()

**Purpose**: Load images for evaluation with automatic retry

```python
def __getitem__(self, index):
    """Load image with automatic retry on errors (evaluation mode)."""
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            ann = self.annotation[index]
            
            # Load image
            image_path = os.path.join(self.vis_root, ann["image"])
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            image = self.vis_processor(image)
            
            return {
                "image": image,
                "image_id": ann.get("image_id", index),
            }
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                # Retry with random index
                index = random.randint(0, len(self.annotation) - 1)
            else:
                logger.error(
                    f"Failed to load CaptionEvalDataset sample after {max_retries} retries. "
                    f"Index: {index}, Error: {str(e)}"
                )
                raise RuntimeError(
                    f"Cannot load valid image sample after {max_retries} retries"
                ) from e
    
    raise RuntimeError("Unexpected error in CaptionEvalDataset.__getitem__ retry loop")
```

---

## CaptionInstructDataset.__getitem__()

**Purpose**: Create instruction-tuned captions from parent dataset

```python
def __getitem__(self, index):
    """Get instruction-tuned caption from parent dataset."""
    # Parent dataset guarantees valid return (never None), so no retry needed here
    # Just pass through with instruction-specific formatting
    data = super().__getitem__(index)
    
    # Add instruction prefix
    data["text_input"] = f"Caption: {data['text_input']}"
    
    return data
```

---

## File 2: lavis/datasets/datasets/coco_caption_datasets.py

### Imports Added
```python
import random
import logging

logger = logging.getLogger(__name__)
```

---

## COCOCapEvalDataset.__getitem__()

**Purpose**: Load COCO evaluation images with automatic retry

```python
def __getitem__(self, index):
    """Load COCO evaluation image with automatic retry."""
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            ann = self.annotation[index]
            
            # Load image
            image_path = os.path.join(self.vis_root, ann["image"])
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            image = self.vis_processor(image)
            
            # Extract image ID from annotation
            image_id = ann.get("image_id", index)
            
            return {
                "image": image,
                "image_id": image_id,
                "instance_id": index,
            }
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                # Retry with random index
                index = random.randint(0, len(self.annotation) - 1)
            else:
                logger.error(
                    f"Failed to load COCOCapEvalDataset sample after {max_retries} retries. "
                    f"Index: {index}, Error: {str(e)}"
                )
                raise RuntimeError(
                    f"Cannot load valid COCO evaluation sample after {max_retries} retries"
                ) from e
    
    raise RuntimeError("Unexpected error in COCOCapEvalDataset.__getitem__ retry loop")
```

---

## NoCapsEvalDataset.__getitem__()

**Purpose**: Load NoCaps evaluation images with automatic retry

```python
def __getitem__(self, index):
    """Load NoCaps evaluation image with automatic retry."""
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            ann = self.annotation[index]
            
            # Load image
            image_path = os.path.join(self.vis_root, ann["image"])
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            image = self.vis_processor(image)
            
            # Extract image ID from annotation
            image_id = ann.get("img_id", index)
            
            return {
                "image": image,
                "image_id": image_id,
                "instance_id": index,
            }
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                # Retry with random index
                index = random.randint(0, len(self.annotation) - 1)
            else:
                logger.error(
                    f"Failed to load NoCapsEvalDataset sample after {max_retries} retries. "
                    f"Index: {index}, Error: {str(e)}"
                )
                raise RuntimeError(
                    f"Cannot load valid NoCaps evaluation sample after {max_retries} retries"
                ) from e
    
    raise RuntimeError("Unexpected error in NoCapsEvalDataset.__getitem__ retry loop")
```

---

## COCOKarpathyDataset.__getitem__()

**Purpose**: Load Karpathy JSON format COCO images with caption validation

```python
def __getitem__(self, index):
    """Load Karpathy JSON COCO sample with automatic retry and caption validation."""
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            ann = self.annotation[index]["images"][0]
            
            # Load image
            image_path = os.path.join(self.vis_root, ann["filename"])
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            image = self.vis_processor(image)
            
            # Get caption
            caption = ann["sentences"][0]["raw"]
            
            # Process caption
            caption = self.text_processor(caption)
            
            # Validate caption is not empty
            if not caption or (isinstance(caption, str) and not caption.strip()):
                raise ValueError(f"Empty caption after processing: {ann['filename']}")
            
            return {
                "image": image,
                "text_input": caption,
                "image_id": ann["imgid"],
            }
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                # Retry with random index
                index = random.randint(0, len(self.annotation) - 1)
            else:
                logger.error(
                    f"Failed to load COCOKarpathyDataset sample after {max_retries} retries. "
                    f"Index: {index}, Error: {str(e)}"
                )
                raise RuntimeError(
                    f"Cannot load valid Karpathy sample after {max_retries} retries"
                ) from e
    
    raise RuntimeError("Unexpected error in COCOKarpathyDataset.__getitem__ retry loop")
```

---

## COCOKarpathyEvalDataset.__getitem__()

**Purpose**: Load Karpathy JSON format evaluation images with multiple captions

```python
def __getitem__(self, index):
    """Load Karpathy JSON evaluation sample with automatic retry."""
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            ann = self.annotation[index]["images"][0]
            
            # Load image
            image_path = os.path.join(self.vis_root, ann["filename"])
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            image = self.vis_processor(image)
            
            # Get all captions
            captions = [s["raw"] for s in ann["sentences"]]
            
            return {
                "image": image,
                "captions": captions,
                "image_id": ann["imgid"],
                "instance_id": index,
            }
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                # Retry with random index
                index = random.randint(0, len(self.annotation) - 1)
            else:
                logger.error(
                    f"Failed to load COCOKarpathyEvalDataset sample after {max_retries} retries. "
                    f"Index: {index}, Error: {str(e)}"
                )
                raise RuntimeError(
                    f"Cannot load valid Karpathy evaluation sample after {max_retries} retries"
                ) from e
    
    raise RuntimeError("Unexpected error in COCOKarpathyEvalDataset.__getitem__ retry loop")
```

---

## Key Implementation Details

### 1. Retry Loop Structure
```python
max_retries = 10
retry_count = 0

while retry_count < max_retries:
    try:
        # Load and return data
        return {...}
    except Exception as e:
        retry_count += 1
        if retry_count < max_retries:
            # Pick new random index
            index = random.randint(0, len(self.annotation) - 1)
        else:
            # Max retries exceeded - raise exception
            logger.error(...)
            raise RuntimeError(...) from e

# Safety: Should never reach here
raise RuntimeError("Unexpected error...")
```

### 2. Logger Integration
```python
logger.error(
    f"Failed to load {ClassName} sample after {max_retries} retries. "
    f"Index: {index}, Error: {str(e)}"
)
```

### 3. Exception Chaining
```python
raise RuntimeError(f"Cannot load valid ... sample after {max_retries} retries") from e
```

### 4. Caption Validation
```python
# For training datasets only (not eval)
if not caption or (isinstance(caption, str) and not caption.strip()):
    raise ValueError(f"Empty caption after processing: {path}")
```

---

## Summary Table

| Dataset Class | Retry? | Caption Validation? | Returns |
|---|---|---|---|
| CaptionDataset | ✅ Yes | ✅ Yes | dict or RuntimeError |
| CaptionEvalDataset | ✅ Yes | ❌ No | dict or RuntimeError |
| CaptionInstructDataset | ❌ No (parent handles) | ✅ Via parent | dict or RuntimeError |
| COCOCapEvalDataset | ✅ Yes | ❌ No | dict or RuntimeError |
| NoCapsEvalDataset | ✅ Yes | ❌ No | dict or RuntimeError |
| COCOKarpathyDataset | ✅ Yes | ✅ Yes | dict or RuntimeError |
| COCOKarpathyEvalDataset | ✅ Yes | ❌ No | dict or RuntimeError |

---

## Testing

```python
# Test 1: Normal loading (should work)
dataset = CaptionDataset(...)
item = dataset[0]
assert item is not None
assert "image" in item
assert "text_input" in item

# Test 2: No None in batch
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32)
for batch in dataloader:
    assert batch["image"] is not None
    assert batch["text_input"] is not None

# Test 3: Exception on all retries (with corrupted data)
# When all 10 retries fail, should raise RuntimeError
try:
    item = corrupted_dataset[0]
    assert False, "Should have raised RuntimeError"
except RuntimeError as e:
    assert "Cannot load valid" in str(e)
    print("✓ Correctly raises RuntimeError on max retries")
```

