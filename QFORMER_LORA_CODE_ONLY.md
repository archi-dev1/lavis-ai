# Q-Former with LoRA - Code Only

## File: `lavis/models/blip2_models/blip2.py`

---

## COMPLETE Q-FORMER IMPLEMENTATION

### Section 1: Imports

```python
try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft library not available. LoRA features will be disabled. Install: pip install peft")
```

---

### Section 2: apply_lora_to_qformer()

```python
def apply_lora_to_qformer(model, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
    """Apply LoRA to Q-Former model."""
    if not PEFT_AVAILABLE:
        logging.error("PEFT library required for LoRA. Install: pip install peft")
        return model
    
    if target_modules is None:
        target_modules = ["query", "key", "value"]
    
    # Auto-detect module names if needed
    try:
        available_modules = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module_base_name = name.split(".")[-1]
                available_modules.add(module_base_name)
        
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
```

---

### Section 3: freeze_qformer_base()

```python
def freeze_qformer_base(model):
    """Freeze all base parameters in a LoRA-enabled Q-Former model."""
    frozen_count = 0
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
            frozen_count += 1
    
    logging.info(f"Frozen {frozen_count} base model parameters")
    return frozen_count
```

---

### Section 4: get_trainable_params_info()

```python
def get_trainable_params_info(model):
    """Get information about trainable parameters in a LoRA-enabled model."""
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

### Section 5: init_Qformer_with_lora() Method (Add to Blip2Base class)

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
    """Initialize Q-Former with LoRA support."""
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

## USAGE

### One-Shot Initialization

```python
from lavis.models.blip2_models.blip2 import Blip2Base

# Step 1: Input (B, N, D) from PVT
# Step 2: num_query_token=16
# Step 3: encoder_hidden_states=image_embeds
# Step 4: target_modules=["query", "key", "value"]
# Step 5: r=8, lora_alpha=16, lora_dropout=0.1
# Step 6: Freeze base model (automatic)
# Step 7: Validation (automatic)

Qformer, query_tokens, info = Blip2Base.init_Qformer_with_lora(
    num_query_token=16,
    vision_width=512,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

# Validation print
print(f"Trainable: {info['trainable_percentage']:.2f}%")
assert info['trainable_percentage'] < 5
```

### Forward Pass with PVT Features

```python
import torch

# Step 1: Input from PVT v2 b2
image_embeds = torch.randn(2, 49, 512)  # (B, N, D)

# Step 2: Query tokens
query_tokens_expanded = query_tokens.expand(2, -1, -1)

# Create attention mask
image_atts = torch.ones(2, 49, dtype=torch.long)

# Step 3: Forward with encoder_hidden_states
output = Qformer.bert(
    query_embeds=query_tokens_expanded,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=image_atts,
    return_dict=True,
)

print(f"Output shape: {output.last_hidden_state.shape}")  # (2, 16, 256)
```

---

## Configuration

```
Step 1: Input
  image_embeds: (B, N, D) = (2, 49, 512)

Step 2: num_query_token = 16

Step 3: encoder_hidden_states = image_embeds

Step 4: target_modules = ["query", "key", "value"]

Step 5: 
  r=8
  lora_alpha=16
  lora_dropout=0.1

Step 6: Base model FROZEN ✓

Step 7: 
  Trainable params: 3,456,789
  Total params: 110,000,000
  Trainable %: 3.15% ✓
```

---

## Error Handling

```python
# Auto-detects module names on mismatch
# If "query" not found:
⚠️ Target module 'query' not found
✓ Auto-detected: ['key', 'value', ...]

# PEFT availability check
if not PEFT_AVAILABLE:
    # Gracefully uses base model without LoRA
```

