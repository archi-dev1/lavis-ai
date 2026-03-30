# Dataset __getitem__ Retry Mechanism - Testing & Validation Guide

## 🧪 Testing Strategy

After implementing the retry mechanism, validate with these tests:

### Test 1: Normal Dataset Loading (Sanity Check)

```python
from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.processors import load_processor
import os

# Setup
vis_processor = load_processor(type="blip_image_eval", cfg=None)
text_processor = load_processor(type="blip_text", cfg=None)
vis_root = "path/to/coco/images"
ann_paths = ["path/to/annotations.json"]

# Test 1: Create dataset
try:
    dataset = CaptionDataset(vis_processor, text_processor, vis_root, ann_paths)
    print("✓ Dataset created successfully")
except Exception as e:
    print(f"✗ Dataset creation failed: {e}")

# Test 2: Get single item (should never return None)
try:
    item = dataset[0]
    assert item is not None, "Dataset returned None"
    assert "image" in item, "Missing 'image' key"
    assert "text_input" in item, "Missing 'text_input' key"
    print("✓ Single item loaded successfully")
    print(f"  - Image shape: {item['image'].shape if hasattr(item['image'], 'shape') else 'tensor'}")
    print(f"  - Caption: {item['text_input'][:50]}...")
except Exception as e:
    print(f"✗ Single item loading failed: {e}")

# Test 3: Get multiple items (test randomness)
try:
    indices = [0, 1, 5, 10, 15]
    items = [dataset[i] for i in indices]
    
    for item in items:
        assert item is not None
        assert "image" in item
    
    print(f"✓ Loaded {len(items)} items successfully (no None)")
except Exception as e:
    print(f"✗ Multiple item loading failed: {e}")
```

---

### Test 2: DataLoader Batch Loading (No None Contamination)

```python
from torch.utils.data import DataLoader
import torch

# Setup
dataset = CaptionDataset(vis_processor, text_processor, vis_root, ann_paths)

# Custom collate to verify no None
def collate_fn(batch):
    """Verify no None values in batch"""
    images = []
    captions = []
    
    for item in batch:
        # This should never happen with new retry mechanism
        if item is None:
            print(f"ERROR: None found in batch!")
            raise ValueError("None item in batch - retry mechanism failed")
        
        images.append(item["image"])
        captions.append(item["text_input"])
    
    return {
        "image": torch.stack(images, dim=0),
        "text_input": captions,
    }

# Test 1: Create DataLoader
try:
    dataloader = DataLoader(
        dataset, 
        batch_size=32,
        num_workers=0,  # Start with single worker
        collate_fn=collate_fn,
        shuffle=True
    )
    print("✓ DataLoader created successfully")
except Exception as e:
    print(f"✗ DataLoader creation failed: {e}")

# Test 2: Iterate through batches
try:
    num_batches = min(5, len(dataloader))  # Check first 5 batches
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        # Verify no None
        assert batch["image"] is not None
        assert batch["text_input"] is not None
        assert len(batch["text_input"]) == batch["image"].shape[0]
        
        print(f"✓ Batch {batch_idx+1}: {len(batch['text_input'])} samples, "
              f"images shape: {batch['image'].shape}")
    
    print(f"✓ All {num_batches} batches loaded without None values")
    
except Exception as e:
    print(f"✗ Batch loading failed: {e}")
```

---

### Test 3: Verify Retry Mechanism Triggers (Inject Failures)

```python
import random
from unittest.mock import patch, MagicMock
from PIL import Image

# Test when Image.open fails
def test_image_load_retry():
    """Verify retry mechanism when image loading fails"""
    dataset = CaptionDataset(vis_processor, text_processor, vis_root, ann_paths)
    
    # Mock Image.open to fail sometimes
    original_open = Image.open
    call_count = [0]
    
    def failing_open(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:  # Fail first 2 attempts
            raise IOError(f"Simulated image load failure (attempt {call_count[0]})")
        return original_open(*args, **kwargs)
    
    with patch('PIL.Image.open', side_effect=failing_open):
        try:
            item = dataset[0]
            assert item is not None
            assert call_count[0] > 2  # Should have retried
            print(f"✓ Retry mechanism triggered: {call_count[0]} total attempts for 1 sample")
        except Exception as e:
            print(f"✗ Retry mechanism failed: {e}")
```

---

### Test 4: Max Retries Exhaustion

```python
from PIL import Image

def test_max_retries_exhaustion():
    """Verify RuntimeError raised when max retries exceeded"""
    dataset = CaptionDataset(vis_processor, text_processor, vis_root, ann_paths)
    
    # Mock Image.open to always fail
    def always_fail(*args, **kwargs):
        raise IOError("Simulated permanent image load failure")
    
    with patch('PIL.Image.open', side_effect=always_fail):
        try:
            item = dataset[0]
            print(f"✗ Should have raised RuntimeError after max retries")
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            if "Cannot load valid" in str(e) and "10 retries" in str(e):
                print(f"✓ Correctly raised RuntimeError after max retries")
                print(f"  Error message: {str(e)[:80]}...")
            else:
                print(f"✗ Wrong error message: {e}")
        except Exception as e:
            print(f"✗ Wrong exception type: {type(e).__name__}: {e}")
```

---

### Test 5: Caption Validation

```python
def test_caption_validation():
    """Verify empty caption check works"""
    dataset = CaptionDataset(vis_processor, text_processor, vis_root, ann_paths)
    
    # Mock text_processor to return empty string sometimes
    original_processor = dataset.text_processor
    call_count = [0]
    
    def sometimes_empty(text):
        call_count[0] += 1
        if call_count[0] <= 2:  # Return empty for first 2 attempts
            return ""
        return original_processor(text)
    
    with patch.object(dataset, 'text_processor', side_effect=sometimes_empty):
        try:
            item = dataset[0]
            assert item is not None
            assert item["text_input"]  # Should not be empty
            print(f"✓ Caption validation triggered: {call_count[0]} attempts")
        except Exception as e:
            print(f"✗ Caption validation failed: {e}")
```

---

### Test 6: Multi-Worker DataLoader

```python
def test_multiworker_dataloader():
    """Test with multiple workers (parallel loading)"""
    dataset = CaptionDataset(vis_processor, text_processor, vis_root, ann_paths)
    
    try:
        # Note: num_workers > 0 may have issues on some systems
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            num_workers=2,
            shuffle=True,
            pin_memory=True
        )
        
        # Load a few batches
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
            
            assert batch["image"] is not None
            print(f"✓ Batch {batch_idx+1} (num_workers=2): {len(batch['text_input'])} samples")
        
        print("✓ Multi-worker DataLoader works correctly")
    
    except Exception as e:
        print(f"✗ Multi-worker DataLoader failed: {e}")
        print("  (This may be expected on some systems; try num_workers=0)")
```

---

### Test 7: Different Dataset Classes

```python
def test_all_dataset_classes():
    """Test all modified dataset classes"""
    from lavis.datasets.datasets.caption_datasets import (
        CaptionDataset, CaptionEvalDataset, CaptionInstructDataset
    )
    from lavis.datasets.datasets.coco_caption_datasets import (
        COCOCapEvalDataset, NoCapsEvalDataset,
        COCOKarpathyDataset, COCOKarpathyEvalDataset
    )
    
    classes_to_test = [
        ("CaptionDataset", CaptionDataset),
        ("CaptionEvalDataset", CaptionEvalDataset),
        ("CaptionInstructDataset", CaptionInstructDataset),
        ("COCOCapEvalDataset", COCOCapEvalDataset),
        ("NoCapsEvalDataset", NoCapsEvalDataset),
        ("COCOKarpathyDataset", COCOKarpathyDataset),
        ("COCOKarpathyEvalDataset", COCOKarpathyEvalDataset),
    ]
    
    for class_name, dataset_class in classes_to_test:
        try:
            # Note: Requires appropriate annotation files
            dataset = dataset_class(vis_processor, text_processor, vis_root, ann_paths)
            item = dataset[0]
            
            assert item is not None, f"{class_name} returned None"
            print(f"✓ {class_name:30s} - OK")
        
        except FileNotFoundError:
            print(f"⚠ {class_name:30s} - SKIP (annotation not found)")
        except Exception as e:
            print(f"✗ {class_name:30s} - FAIL: {str(e)[:50]}")
```

---

## 📊 Complete Test Suite

```python
import logging

# Enable logging to see retry messages
logging.basicConfig(level=logging.INFO)
logging.getLogger('lavis.datasets.datasets.caption_datasets').setLevel(logging.DEBUG)
logging.getLogger('lavis.datasets.datasets.coco_caption_datasets').setLevel(logging.DEBUG)

def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("DATASET __getitem__ RETRY MECHANISM - VALIDATION TESTS")
    print("="*60)
    
    tests = [
        ("Normal Loading", test_normal_loading),
        ("DataLoader Batching", test_dataloader),
        ("Retry Mechanism", test_retry),
        ("Max Retries Exhaustion", test_max_retries),
        ("Caption Validation", test_caption_validation),
        ("Multi-Worker DataLoader", test_multiworker),
        ("All Dataset Classes", test_all_classes),
    ]
    
    results = []
    for test_name, test_fn in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 60)
        try:
            test_fn()
            results.append((test_name, "PASS"))
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append((test_name, "FAIL"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {test_name:30s} - {result}")
    
    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
```

---

## 🔍 Debugging Retry Mechanism

### Check Retry Logging

```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable dataset logging
logging.getLogger('lavis.datasets.datasets.caption_datasets').setLevel(logging.DEBUG)
logging.getLogger('lavis.datasets.datasets.coco_caption_datasets').setLevel(logging.DEBUG)

# Now load dataset - you'll see retry messages
dataset = CaptionDataset(vis_processor, text_processor, vis_root, ann_paths)

# This will show error messages if retries occur
item = dataset[0]
```

### Example Log Output

```
2024-01-15 10:30:45 - lavis.datasets - ERROR - Failed to load sample after 3 retries. Last error: [Errno 2] No such file or directory: '/path/to/missing/image.jpg'
```

### Capture Exception Info

```python
try:
    dataset = CaptionDataset(...)
    item = dataset[100]  # Try problematic index
except RuntimeError as e:
    print(f"Caught RuntimeError after retries:")
    print(f"  Message: {e}")
    print(f"  Cause: {e.__cause__}")  # Original exception
```

---

## ✅ Expected Results

### Before Retry Mechanism
```
dataset[0] → None (if image loading fails)
Batch loading → Contains None values
Downstream code → Crashes or skips None items
```

### After Retry Mechanism
```
dataset[0] → Valid dict or RuntimeError
Batch loading → All items guaranteed valid
Downstream code → Never encounters None
```

---

## 📝 Checklist

- [ ] Normal dataset loading works
- [ ] DataLoader batches load without None
- [ ] Single items never return None
- [ ] Retry mechanism triggers on failures
- [ ] Max retries raises RuntimeError
- [ ] Caption validation works
- [ ] All 6 dataset classes work
- [ ] Logging shows retry attempts
- [ ] Multi-worker DataLoader works
- [ ] No syntax errors in codebase

---

## 🚀 Production Validation

Once tests pass, validate in production:

```python
# 1. Load with real data
dataset = CaptionDataset(vis_processor, text_processor, vis_root, ann_paths)

# 2. Create DataLoader for training
train_loader = DataLoader(
    dataset, 
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True
)

# 3. Train with no None contamination
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        captions = batch['text_input']
        
        # Process batch - guaranteed no None values
        loss = model(images, captions)
        loss.backward()
        optimizer.step()
```

---

## 💡 Troubleshooting

| Issue | Solution |
|-------|----------|
| `RuntimeError: Cannot load valid sample` | All 10 retries failed - check image paths and permissions |
| `AttributeError: module has no attribute 'random'` | Missing `import random` - verify imports at top of file |
| `TypeError: unsupported operand type` | Caption validation logic issue - check text_processor output |
| `None in batch` | Retry mechanism not applied - verify file was saved correctly |
| `FileNotFoundError` | Annotation or image path issues - verify dataset paths |

---

## 🎉 Summary

✅ **Test normal loading** - Sanity check  
✅ **Test DataLoader** - Batch safety  
✅ **Test retries** - Mechanism validation  
✅ **Test max retries** - Exception handling  
✅ **Test validation** - Caption checks  
✅ **Test all classes** - Complete coverage  
✅ **Test logging** - Error messages  

**Result**: Comprehensive validation of robust, production-ready dataset implementation!

