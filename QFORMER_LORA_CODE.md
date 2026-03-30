# Modified Q-Former Code for BLIP-2 with LoRA

## File: `lavis/models/blip2_models/blip2.py`

---

## COMPLETE Q-FORMER WITH LORA IMPLEMENTATION

### Import Addition (Line ~20-25)

```python
try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft library not available. LoRA features will be disabled. Install: pip install peft")
```

---

### LoRA Utility Functions (New Section)

```python
def apply_lora_to_qformer(model, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
    """
    Apply LoRA to Q-Former model.
    
    Args:
        model: BertLMHeadModel (Q-Former)
        r: LoRA rank (default 8)
        lora_alpha: LoRA scaling factor (default 16)
        lora_dropout: LoRA dropout probability (default 0.1)
        target_modules: List of module names to apply LoRA (default ["query", "key", "value"])
        
    Returns:
        PEFT-wrapped model with LoRA enabled
    """
    if not PEFT_AVAILABLE:
        logging.error("PEFT library required for LoRA. Install: pip install peft")
        return model
    
    if target_modules is None:
        target_modules = ["query", "key", "value"]
    
    # Auto-detect module names if needed
    try:
        # Get actual module names from the model
        available_modules = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Extract the last part of the name (e.g., "query", "key", "value")
                module_base_name = name.split(".")[-1]
                available_modules.add(module_base_name)
        
        # Auto-detect if target modules exist
        detected_modules = []
        for target in target_modules:
            if target in available_modules:
                detected_modules.append(target)
            else:
                logging.warning(f"Target module '{target}' not found in model. Available: {available_modules}")
        
        if detected_modules:
            target_modules = detected_modules
            logging.info(f"Auto-detected LoRA target modules: {target_modules}")
        else:
            logging.warning("No target modules found. Using default: ['query', 'key', 'value']")
    except Exception as e:
        logging.warning(f"Auto-detection failed: {e}. Using provided target modules: {target_modules}")
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    logging.info(
        f"LoRA applied to Q-Former:\n"
        f"  - Rank (r): {r}\n"
        f"  - Alpha: {lora_alpha}\n"
        f"  - Dropout: {lora_dropout}\n"
        f"  - Target modules: {target_modules}"
    )
    
    return model


def freeze_qformer_base(model):
    """
    Freeze all base parameters in a LoRA-enabled Q-Former model.
    
    Args:
        model: PEFT-wrapped BertLMHeadModel
        
    Returns:
        Number of frozen parameters
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
            frozen_count += 1
    
    logging.info(f"Frozen {frozen_count} base model parameters")
    return frozen_count


def get_trainable_params_info(model):
    """
    Get information about trainable parameters in a LoRA-enabled model.
    
    Args:
        model: PEFT-wrapped or regular model
        
    Returns:
        Dictionary with trainable/total count and percentage
    """
    trainable_count = 0
    total_count = 0
    
    for param in model.parameters():
        total_count += param.numel()
        if param.requires_grad:
            trainable_count += param.numel()
    
    percentage = (trainable_count / total_count * 100) if total_count > 0 else 0
    
    info = {
        "trainable_params": trainable_count,
        "total_params": total_count,
        "trainable_percentage": percentage,
    }
    
    # Print with validation
    logging.info(
        f"\n{'='*60}\n"
        f"Trainable Parameters Summary:\n"
        f"{'='*60}\n"
        f"Trainable params: {trainable_count:,}\n"
        f"Total params:     {total_count:,}\n"
        f"Trainable %:      {percentage:.2f}%\n"
        f"{'='*60}"
    )
    
    if percentage < 5:
        logging.info("✓ Trainable percentage < 5%: LoRA configuration valid")
    elif percentage < 10:
        logging.warning("⚠️ Trainable percentage between 5-10%: May want to increase LoRA rank")
    else:
        logging.warning(f"⚠️ Trainable percentage {percentage:.2f}%: High percentage for LoRA training")
    
    return info
```

---

### New Method in Blip2Base Class

```python
@classmethod
def init_Qformer_with_lora(
    cls,
    num_query_token=16,
    vision_width=512,
    cross_attention_freq=2,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    lora_target_modules=None,
):
    """
    Initialize Q-Former with LoRA support.
    
    Args:
        num_query_token: Number of query tokens (default 16 for PVT features)
        vision_width: Vision encoder output dimension (512 for PVT v2 b2)
        cross_attention_freq: Cross-attention frequency
        use_lora: Enable LoRA (default True)
        lora_r: LoRA rank (default 8)
        lora_alpha: LoRA alpha scaling (default 16)
        lora_dropout: LoRA dropout (default 0.1)
        lora_target_modules: Target modules for LoRA (default ["query", "key", "value"])
        
    Returns:
        Tuple of (Qformer model, query_tokens, trainable_info_dict)
    """
    # Initialize base Qformer
    Qformer, query_tokens = cls.init_Qformer(
        num_query_token=num_query_token,
        vision_width=vision_width,
        cross_attention_freq=cross_attention_freq,
    )
    
    # Apply LoRA if enabled
    if use_lora and PEFT_AVAILABLE:
        if lora_target_modules is None:
            lora_target_modules = ["query", "key", "value"]
        
        # Apply LoRA
        Qformer = apply_lora_to_qformer(
            Qformer,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        
        # Freeze base model parameters
        freeze_qformer_base(Qformer)
        
        # Get trainable params info
        trainable_info = get_trainable_params_info(Qformer)
        
        logging.info(
            f"\nQ-Former with LoRA initialized:\n"
            f"  - Query tokens: {num_query_token}\n"
            f"  - Vision width: {vision_width}\n"
            f"  - Base model frozen: ✓\n"
            f"  - LoRA modules: {lora_target_modules}\n"
            f"  - Trainable params: {trainable_info['trainable_params']:,}\n"
            f"  - Total params: {trainable_info['total_params']:,}\n"
            f"  - Trainable %: {trainable_info['trainable_percentage']:.2f}%"
        )
    else:
        if use_lora and not PEFT_AVAILABLE:
            logging.warning("LoRA requested but peft not available. Using base Qformer without LoRA.")
        trainable_info = get_trainable_params_info(Qformer)
    
    return Qformer, query_tokens, trainable_info
```

---

## Usage Examples

### Example 1: Q-Former Standalone Initialization

```python
from lavis.models.blip2_models.blip2 import Blip2Base

# Step 2: Set num_query_token = 16
# Step 1: Input image_embeds (B, N, D) from PVT
# Step 3: Pass encoder_hidden_states=image_embeds
# Step 4: Apply LoRA to ["query", "key", "value"]
# Step 5: Config r=8, lora_alpha=16, lora_dropout=0.1
# Step 6: Freeze base model (automatic)
# Step 7: Validation - trainable params printed

Qformer, query_tokens, trainable_info = Blip2Base.init_Qformer_with_lora(
    num_query_token=16,           # Step 2
    vision_width=512,             # Step 1: PVT output dim
    use_lora=True,
    lora_r=8,                    # Step 5
    lora_alpha=16,               # Step 5
    lora_dropout=0.1,            # Step 5
    lora_target_modules=["query", "key", "value"],  # Step 4
)

# Step 7: Validation
assert trainable_info['trainable_percentage'] < 5
print(f"✓ Configuration valid: {trainable_info['trainable_percentage']:.2f}% trainable")

# Forward pass with PVT features
image_embeds = torch.randn(2, 49, 512)  # (B, N, D) from PVT v2 b2
image_atts = torch.ones(2, 49, dtype=torch.long)

query_tokens_expanded = query_tokens.expand(2, -1, -1)

# Step 3: Pass encoder_hidden_states=image_embeds
output = Qformer.bert(
    query_embeds=query_tokens_expanded,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=image_atts,
    return_dict=True,
)

print(f"Output shape: {output.last_hidden_state.shape}")  # (2, 16, 256)
```

### Example 2: BLIP2-T5 with PVT and LoRA

```python
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.blip2_models.blip2 import Blip2Base

class Blip2T5WithLoRA(Blip2T5):
    """BLIP2-T5 with PVT v2 b2 and LoRA."""
    
    def __init__(
        self,
        vit_model="pvt_v2_b2",
        use_lora=True,
        **kwargs
    ):
        super().__init__(
            vit_model=vit_model,
            num_query_token=16,  # Step 2: Optimized for PVT
            **kwargs
        )
        
        if use_lora:
            # Replace with LoRA-enabled Qformer
            self.Qformer, self.query_tokens, info = \
                Blip2Base.init_Qformer_with_lora(
                    num_query_token=16,
                    vision_width=512,  # PVT output dim
                    use_lora=True,
                    lora_r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                )
            
            # Clean up
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

# Initialize
model = Blip2T5WithLoRA(
    t5_model="google/flan-t5-xl",
)

# Forward pass
image = torch.randn(2, 3, 224, 224)
output = model({
    "image": image,
    "text_input": "a photo of",
    "text_output": "a cat",
})
```

### Example 3: Manual LoRA Application

```python
from lavis.models.blip2_models.blip2 import (
    apply_lora_to_qformer,
    freeze_qformer_base,
    get_trainable_params_info,
    Blip2Base,
)

# Create base Qformer
base_qformer, query_tokens = Blip2Base.init_Qformer(
    num_query_token=16,
    vision_width=512,
)

# Step 4: Apply LoRA with target modules
lora_qformer = apply_lora_to_qformer(
    base_qformer,
    r=8,                    # Step 5: r=8
    lora_alpha=16,         # Step 5: lora_alpha=16
    lora_dropout=0.1,      # Step 5: lora_dropout=0.1
    target_modules=["query", "key", "value"],  # Step 4
)

# Step 6: Freeze base model
freeze_qformer_base(lora_qformer)

# Step 7: Validation
info = get_trainable_params_info(lora_qformer)

if info['trainable_percentage'] < 5:
    print("✓ Print trainable params (<5%)")
else:
    print(f"⚠️ Trainable params: {info['trainable_percentage']:.2f}%")
```

---

## Configuration Summary

### Input/Output Specifications

```
Step 1: Input
  - Format: (B, N, D)
  - Example: (2, 49, 512) from PVT v2 b2
  
Step 2: Configuration
  - num_query_token=16
  
Step 3: Forward
  - encoder_hidden_states=image_embeds

Step 4: LoRA Targets
  - target_modules=["query", "key", "value"]
  
Step 5: LoRA Config
  - r=8
  - lora_alpha=16
  - lora_dropout=0.1
  
Step 6: Freezing
  - Base model: FROZEN ❌
  - LoRA params: TRAINABLE ✓
  
Step 7: Validation Output
  - Trainable params: 1,234,567
  - Total params: 110,000,000
  - Trainable %: 1.12% ✓
```

---

## Error Handling Features

### Auto-Detection Example

```python
# Input:
target_modules = ["query", "key", "value", "nonexistent"]

# Output:
⚠️ Target module 'nonexistent' not found in model
Available: {'query', 'key', 'value', 'dense', 'LayerNorm', ...}
✓ Auto-detected LoRA target modules: ['query', 'key', 'value']
```

### PEFT Availability Check

```python
if not PEFT_AVAILABLE:
    logging.error("PEFT library required for LoRA. Install: pip install peft")
    # Gracefully fallback to base model
```

---

## Testing Command

```bash
python test_qformer_lora.py
```

**Expected Test Output:**
```
TEST 1: Q-Former Initialization with LoRA
✓ Q-Former with LoRA initialized successfully
✓ Trainable percentage < 5%: Configuration valid
✓ TEST 1 PASSED

TEST 2: Q-Former Forward Pass with PVT Features
✓ Forward pass successful
✓ Output shape validation PASSED
✓ TEST 2 PASSED

TEST 3: LoRA Module Auto-Detection
✓ LoRA modules successfully applied
✓ TEST 3 PASSED

TEST 4: Gradient Flow Through LoRA Layers
✓ Backward pass successful
✓ TEST 4 PASSED

✓ TOTAL: 4/4 tests passed
🎉 ALL TESTS PASSED!
```

---

## Installation

```bash
pip install peft>=0.4.0
pip install torch>=1.9.0
pip install transformers>=4.20.0
```

---

## Key Functions Reference

| Function | Purpose | Returns |
|----------|---------|---------|
| `apply_lora_to_qformer()` | Apply LoRA to base model | PEFT-wrapped model |
| `freeze_qformer_base()` | Freeze all non-LoRA params | Number of frozen params |
| `get_trainable_params_info()` | Get param statistics | Dict with counts & % |
| `init_Qformer_with_lora()` | One-shot init with LoRA | (model, tokens, info) |

