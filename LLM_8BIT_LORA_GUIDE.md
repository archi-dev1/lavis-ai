# LLM 8-bit LoRA Integration Guide

## Overview

This guide explains how to use the LLM 8-bit quantization with LoRA fine-tuning integration in BLIP-2. This allows you to efficiently fine-tune large language models (LLMs) by:

1. **Loading with 8-bit quantization** - Reduces memory usage from FP32 to INT8 (4x reduction)
2. **Applying LoRA** - Low-Rank Adaptation for parameter-efficient fine-tuning
3. **Freezing base weights** - Only LoRA adapters are trainable (~0.8-2% of parameters)
4. **Graceful fallback** - Automatically switches to FP16 if CUDA unavailable

## Quick Start

### Installation

```bash
# Core dependencies
pip install torch transformers peft

# Optional: For 8-bit quantization (requires CUDA)
pip install bitsandbytes

# Optional: For distributed training
pip install torch-distributed-package
```

### Basic Usage

```python
from lavis.models.blip2_models.blip2 import load_llm_8bit_with_lora

# Load T5 with 8-bit LoRA
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="google/flan-t5-base",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

print(f"Model loaded. Trainable params: {info['trainable_percentage']:.2f}%")
```

## API Reference

### `load_llm_8bit_with_lora()`

Main entry point for loading LLM with 8-bit quantization and LoRA.

**Signature:**
```python
def load_llm_8bit_with_lora(
    model_name: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[List[str]] = None,
    load_in_8bit: bool = True,
    device_map: str = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | Required | HuggingFace model ID (e.g., "google/flan-t5-base", "meta-llama/Llama-2-7b") |
| `lora_r` | int | 8 | LoRA rank. Smaller (4-8) for small models, larger (16-32) for large models |
| `lora_alpha` | int | 16 | LoRA scaling factor. Typically 2×rank for stability |
| `lora_dropout` | float | 0.1 | Dropout probability in LoRA adapters |
| `lora_target_modules` | List[str] | None | Modules to apply LoRA. Auto-detected if None |
| `load_in_8bit` | bool | True | Enable 8-bit quantization (requires CUDA, falls back to fp16) |
| `device_map` | str | "auto" | Device mapping strategy for distributed loading |

**Returns:**

Tuple of (model, tokenizer, info_dict):

```python
{
    'trainable_params': int,        # Number of trainable parameters
    'total_params': int,            # Total parameters in model
    'trainable_percentage': float,  # Percentage of trainable params
    'model_dtype': str,             # Loaded precision (int8, fp16, fp32)
    'lora_config': LoraConfig,      # LoRA configuration object
    'target_modules': List[str],    # Modules with LoRA adapters
}
```

**Examples:**

```python
# Example 1: T5 Base Model
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="google/flan-t5-base",
    lora_r=8,
    lora_alpha=16,
)

# Example 2: Large T5 Model with Higher Rank
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="google/flan-t5-xl",
    lora_r=16,
    lora_alpha=32,
)

# Example 3: LLaMA with Custom Target Modules
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="meta-llama/Llama-2-7b",
    lora_r=8,
    lora_target_modules=["q_proj", "v_proj"],  # Explicit targets
)

# Example 4: No 8-bit (FP16 only)
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="gpt2-medium",
    load_in_8bit=False,
)
```

### `apply_lora_to_llm()`

Low-level function to apply LoRA to an already-loaded model.

**Signature:**
```python
def apply_lora_to_llm(
    model: PreTrainedModel,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    task_type: Optional[str] = None,
) -> Tuple[PreTrainedModel, LoraConfig, List[str]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | PreTrainedModel | Required | Model to apply LoRA to |
| `lora_r` | int | 8 | LoRA rank |
| `lora_alpha` | int | 16 | LoRA scaling factor |
| `lora_dropout` | float | 0.1 | Dropout in LoRA |
| `target_modules` | List[str] | None | Modules to target. Auto-detected if None. |
| `task_type` | str | None | Task type ("CAUSAL_LM", "SEQ_2_SEQ_LM"). Auto-detected if None. |

**Returns:**

Tuple of (model_with_lora, lora_config, target_modules_used)

### `freeze_llm_base()`

Freezes all base LLM parameters, keeping only LoRA adapters trainable.

**Signature:**
```python
def freeze_llm_base(model: PreTrainedModel) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | PreTrainedModel | Model with LoRA adapters |

**Effect:**

- Disables gradients for all non-LoRA parameters
- Enables gradients for LoRA modules (default from PEFT)
- Inverse operation: `model.train()` enables training

### `get_trainable_params_info()`

Validates parameter count and returns training statistics.

**Signature:**
```python
def get_trainable_params_info(model: PreTrainedModel) -> Dict[str, Any]
```

**Returns:**

```python
{
    'trainable_params': int,        # Number of trainable parameters
    'total_params': int,            # Total parameters
    'trainable_percentage': float,  # Percentage (should be <5% for LoRA)
}
```

## Configuration Guide

### LoRA Rank Selection

The LoRA rank controls the capacity of the adapters:

```python
# Small Models (7B parameters or less)
# 4-8 parameters per rank per layer
lora_r = 8   # Minimal overhead, ~0.1-0.5% trainable

# Medium-Large Models (13-30B)
# 8-16 parameters per rank per layer  
lora_r = 16  # Balanced, ~0.5-1.5% trainable

# Large Models (70B+)
# 16-32 parameters per rank per layer
lora_r = 32  # Higher capacity, ~1-3% trainable
```

### Alpha Scaling

The alpha parameter controls LoRA scaling during initialization:

```python
# Alpha = 2 × rank (recommended)
lora_alpha = 2 * lora_r

# This means:
lora_r = 8     → lora_alpha = 16
lora_r = 16    → lora_alpha = 32
lora_r = 32    → lora_alpha = 64
```

**Why 2×rank?**
- Provides stable initialization of LoRA weights
- Prevents exploding/vanishing gradients
- Maintains consistent learning dynamics across different ranks

### Model Type Auto-Detection

The function automatically detects the model type and applies appropriate LoRA targets:

```python
# T5 Models (google/flan-t5-*, t5-*)
# Target modules: ["q", "v"] (query and value in T5 attention)
model_t5 = load_llm_8bit_with_lora("google/flan-t5-large")

# Causal Language Models (meta-llama/Llama-*, gpt*, mistral*)
# Target modules: ["q_proj", "v_proj"]
model_llama = load_llm_8bit_with_lora("meta-llama/Llama-2-7b")

# Seq2Seq Models (facebook/bart-*, google/pegasus-*)
# Target modules: auto-detected based on architecture
model_bart = load_llm_8bit_with_lora("facebook/bart-base")
```

## Error Handling

### Issue: `ModuleNotFoundError: No module named 'bitsandbytes'`

**Symptom:**
```
ModuleNotFoundError: No module named 'bitsandbytes'
```

**Solution:**

The function automatically falls back to FP16 loading. To enable 8-bit:

```bash
pip install bitsandbytes
```

**Note:** Requires CUDA-enabled GPU. For CPU-only environments, FP16 is the fallback.

### Issue: `CUDA out of memory` when loading with 8-bit

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution 1: Reduce Model Size**
```python
# Use smaller model
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="google/flan-t5-base",  # base instead of large
)
```

**Solution 2: Increase LoRA Rank Gradually**
```python
# Start with small rank, increase if OOM persists
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="meta-llama/Llama-2-7b",
    lora_r=4,  # Smaller rank
)
```

### Issue: Trainable Parameters > 5%

**Symptom:**
```
WARNING: Trainable percentage: 8.5% (expected <5%)
```

**Solution: Increase LoRA Rank or Reduce Model Scope**

```python
# Option 1: Reduce LoRA rank
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="model_with_custom_head",
    lora_r=4,  # Smaller rank
)

# Option 2: Specify only necessary target modules
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="meta-llama/Llama-2-7b",
    lora_target_modules=["q_proj"],  # Only query, not value
)
```

## Integration Examples

### Example 1: Fine-tune T5 for Summarization

```python
from lavis.models.blip2_models.blip2 import load_llm_8bit_with_lora
import torch

# Load model with 8-bit LoRA
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="google/flan-t5-base",
    lora_r=8,
)

print(f"Model loaded. Trainable: {info['trainable_percentage']:.2f}%")

# Prepare input
text = "Summarize: Machine learning is..."
inputs = tokenizer(text, return_tensors="pt")

# Forward pass (model in eval mode initially)
model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)

print("Summary:", tokenizer.decode(outputs[0]))

# Training mode
model.train()
# ... training loop ...
```

### Example 2: Integration with BLIP-2

```python
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    load_llm_8bit_with_lora,
)

# Initialize BLIP-2 with 8-bit LLM
class Blip2WithEfficientLLM(Blip2Base):
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        model = cls()
        
        # Load vision encoder
        model.visual_encoder = ...  # Your vision encoder
        
        # Load Q-Former
        model.qformer = ...  # Your Q-Former
        
        # Load LLM with 8-bit LoRA
        model.llm, model.tokenizer, llm_info = load_llm_8bit_with_lora(
            model_name="google/flan-t5-base",
            lora_r=8,
        )
        
        print(f"LLM Info: {llm_info}")
        return model

# Use the model
blip2 = Blip2WithEfficientLLM.from_pretrained("blip2-flan-t5-base")
```

### Example 3: Multi-GPU Training

```python
from lavis.models.blip2_models.blip2 import load_llm_8bit_with_lora
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

# Load with distributed device mapping
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="meta-llama/Llama-2-7b",
    device_map="auto",  # Automatic multi-GPU distribution
    lora_r=16,
)

# Wrap for distributed training
model = DistributedDataParallel(model, find_unused_parameters=True)

# Training setup
optimizer = optim.AdamW(model.parameters(), lr=2e-4)

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Memory Comparison

### T5-Base (220M parameters)

| Method | Memory (GB) | Trainable % | Relative |
|--------|------------|------------|----------|
| FP32 (full) | 0.88 | 100% | 1.0x |
| FP16 (full) | 0.44 | 100% | 0.5x |
| 8-bit (full) | 0.22 | 100% | 0.25x |
| 8-bit + LoRA | 0.22 | 0.6% | 0.25x |

### Benefits

- **8-bit:** 4x memory reduction (0.88 → 0.22 GB)
- **LoRA:** Reduces parameters to 0.6% (18K trainable params)
- **Combined:** Fit fine-tuning on consumer GPUs (8GB VRAM)

## Performance Tips

1. **Use smaller model size first** to validate your approach
2. **Start with rank=8** and increase if needed
3. **Monitor trainable params** - should be <5%
4. **Enable gradient checkpointing** for larger models
5. **Use mixed precision training** (already enabled)

## Advanced Configuration

### Custom Target Modules

```python
# Only fine-tune query projection
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="meta-llama/Llama-2-7b",
    lora_target_modules=["q_proj"],  # Only query
)

# Fine-tune query and value
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="meta-llama/Llama-2-7b",
    lora_target_modules=["q_proj", "v_proj"],
)
```

### Disable 8-bit for Debugging

```python
# Use FP16 for compatibility testing
model, tokenizer, info = load_llm_8bit_with_lora(
    model_name="google/flan-t5-base",
    load_in_8bit=False,  # Use FP16 instead
)
```

## Troubleshooting

### Check Dependencies

```python
from lavis.models.blip2_models.blip2 import BITSANDBYTES_AVAILABLE, PEFT_AVAILABLE

print(f"PEFT: {PEFT_AVAILABLE}")
print(f"Bitsandbytes: {BITSANDBYTES_AVAILABLE}")
print(f"CUDA: {torch.cuda.is_available()}")
```

### Validate Model Info

```python
model, tokenizer, info = load_llm_8bit_with_lora(model_name="...")

# Print detailed info
print(f"Trainable params: {info['trainable_params']:,}")
print(f"Total params: {info['total_params']:,}")
print(f"Trainable %: {info['trainable_percentage']:.2f}%")
print(f"Target modules: {info['target_modules']}")
print(f"Model dtype: {info['model_dtype']}")
```

## Additional Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Bitsandbytes Integration](https://huggingface.co/docs/transformers/quantization)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
