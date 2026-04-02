#!/usr/bin/env python3
"""
BLIP-2 Strict Inference Test
- Uses ONLY model.generate() for captions
- No manual/mock descriptions
- Prints raw token IDs + decoded captions
"""

import sys
import types

# ============================================================
# Step 0: Bypass lavis __init__ import chain (transformers v5)
# ============================================================
for pkg in [
    'lavis', 'lavis.datasets', 'lavis.datasets.builders',
    'lavis.processors', 'lavis.tasks', 'lavis.runners',
    'lavis.models',
]:
    mod = types.ModuleType(pkg)
    mod.__path__ = [pkg.replace('.', '/')]
    mod.__package__ = pkg
    sys.modules[pkg] = mod

sys.path.insert(0, '/workspaces/lavis-ai')

import lavis.common.registry
import lavis.common.utils
from lavis.models.base_model import BaseModel
sys.modules['lavis.models'].BaseModel = BaseModel

# ============================================================
# Now safe to import BLIP-2 modules
# ============================================================
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from lavis.models.blip2_models.blip2_t5 import Blip2T5

print("=" * 70)
print("BLIP-2 STRICT INFERENCE TEST")
print("=" * 70)

# ============================================================
# Step 1: Load model with small T5 (fits in ~4GB RAM)
# ============================================================
print("\n[Step 1] Loading BLIP-2 model (PVT v2 b2 + Q-Former + T5)...")
print("-" * 70)

model = Blip2T5(
    vit_model="pvt_v2_b2",       # Optimized vision encoder (49 patches)
    num_query_token=16,           # Optimized query tokens
    t5_model="google/flan-t5-small",  # Small T5 for CPU inference
    img_size=224,
    drop_path_rate=0,
    use_grad_checkpoint=False,
    vit_precision="fp32",         # CPU needs fp32
    freeze_vit=True,
    prompt="a photo of",
    max_txt_len=32,
)
model.eval()

print("  Vision encoder:  PVT v2 b2 (49 patches, 512 dims)")
print(f"  Q-Former:        {model.query_tokens.shape[1]} query tokens")
print(f"  T5 model:        google/flan-t5-small (d_model=512)")
print(f"  T5 proj:         {model.t5_proj.in_features} -> {model.t5_proj.out_features}")
print(f"  Prompt:          '{model.prompt}'")
print("  Device:          CPU")
print("  Status:          LOADED")

# ============================================================
# Step 2: Create 2 sample images
# ============================================================
print("\n[Step 2] Creating sample images...")
print("-" * 70)

test_images = {}

# Image 1: Warm gradient (red-orange tones)
arr1 = np.zeros((224, 224, 3), dtype=np.uint8)
for i in range(224):
    for j in range(224):
        arr1[i, j] = [min(255, 100 + int(155 * j/224)),
                       int(80 * i/224),
                       int(40 * (1 - j/224))]
test_images["warm_gradient.jpg"] = Image.fromarray(arr1, 'RGB')

# Image 2: Cool pattern (blue-green blocks)
arr2 = np.zeros((224, 224, 3), dtype=np.uint8)
arr2[0:112, 0:112] = [30, 60, 200]     # Blue
arr2[0:112, 112:] = [20, 180, 80]      # Green
arr2[112:, 0:112] = [200, 200, 50]     # Yellow
arr2[112:, 112:] = [180, 40, 40]       # Red
test_images["color_blocks.jpg"] = Image.fromarray(arr2, 'RGB')

for name in test_images:
    print(f"  Created: {name} (224x224 RGB)")

# ============================================================
# Step 3: Convert images to tensors
# ============================================================
print("\n[Step 3] Converting images to tensors...")
print("-" * 70)

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

image_tensors = {}
for name, img in test_images.items():
    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    image_tensors[name] = tensor
    print(f"  {name}: {tensor.shape} (dtype={tensor.dtype})")

# ============================================================
# Step 4 & 5: Pass into model.generate() and decode
# ============================================================
print("\n[Step 4] Running model.generate() (num_beams=3, max_length=20)...")
print("-" * 70)

results = []
with torch.no_grad():
    for name, tensor in image_tensors.items():
        print(f"\nProcessing: {name}")

        # -- Vision encoder --
        with torch.cuda.amp.autocast(enabled=False):
            image_embeds = model.ln_vision(model.visual_encoder(tensor))
        image_embeds = image_embeds.float()
        print(f"  Vision output:  {image_embeds.shape}")

        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long
        )

        # -- Q-Former --
        query_tokens = model.query_tokens.expand(
            image_embeds.shape[0], -1, -1
        )
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        print(f"  Q-Former out:   {query_output.last_hidden_state.shape}")

        # -- T5 projection --
        inputs_t5 = model.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long)
        print(f"  T5 input:       {inputs_t5.shape}")

        # -- Prompt tokens --
        prompt = [model.prompt]
        input_tokens = model.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        )
        encoder_atts = torch.cat(
            [atts_t5, input_tokens.attention_mask], dim=1
        )

        # -- T5 generate (THIS is where captions come from) --
        inputs_embeds = model.t5_model.encoder.embed_tokens(
            input_tokens.input_ids
        )
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        # Cast to bfloat16 for T5
        inputs_embeds = inputs_embeds.to(torch.bfloat16)

        raw_output_ids = model.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=False,
            num_beams=3,
            max_new_tokens=20,
            min_length=1,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_return_sequences=1,
        )

        # Decode
        caption = model.t5_tokenizer.batch_decode(
            raw_output_ids, skip_special_tokens=True
        )[0]

        results.append({
            'name': name,
            'token_ids': raw_output_ids[0].tolist(),
            'caption': caption,
        })

# ============================================================
# Step 6: Print results
# ============================================================
print("\n" + "=" * 70)
print("INFERENCE RESULTS (from model.generate() ONLY)")
print("=" * 70)

for r in results:
    print(f"\nImage:   {r['name']}")
    print(f"Tokens:  {r['token_ids']}")
    print(f"Caption: {r['caption']}")

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)
print(f"  Images processed:        {len(results)}")
print(f"  Captions from generate:  YES (model.t5_model.generate())")
print(f"  Raw tokens shown:        YES")
print(f"  External descriptions:   NO")
all_have_captions = all(len(r['caption']) > 0 for r in results)
all_have_tokens = all(len(r['token_ids']) > 0 for r in results)
print(f"  All captions non-empty:  {'YES' if all_have_captions else 'NO'}")
print(f"  All tokens non-empty:    {'YES' if all_have_tokens else 'NO'}")
if all_have_captions and all_have_tokens:
    print("\n  RESULT: PASS - Inference working, captions generated by model")
else:
    print("\n  RESULT: FAIL")
print("=" * 70)
