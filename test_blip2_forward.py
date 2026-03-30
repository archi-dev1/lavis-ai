#!/usr/bin/env python3
"""
Test BLIP-2 full forward pass with shapes.
"""

import sys
sys.path.insert(0, '/workspaces/lavis-ai')

import torch
import torch.nn as nn
from PIL import Image
import numpy as np


class SimpleLayerNorm(nn.Module):
    """Simple layer norm for testing."""
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
    
    def forward(self, x):
        return self.norm(x)


class SimpleVisionEncoder(nn.Module):
    """Simple vision encoder that mimics output shape."""
    def __init__(self):
        super().__init__()
        # Simulates a vision encoder output: (B, num_patches, hidden_dim)
        self.num_features = 768
    
    def forward(self, x):
        # Input: (B, 3, 224, 224)
        # Output: (B, num_patches, hidden_dim) where num_patches ~= 196
        B = x.shape[0]
        # For a 224x224 image with patch size 16, we get (224/16)^2 = 196 patches
        num_patches = 196
        return torch.randn(B, num_patches, self.num_features, device=x.device)


class SimpleQFormer(nn.Module):
    """Simple Q-Former that mimics the real one."""
    def __init__(self, num_query_tokens=32, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens
        
        # Query tokens: learnable parameters
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, hidden_size))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Simple transformer layer for cross-attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=12, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads=12, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
    
    def forward(self, image_embeds, query_tokens):
        # image_embeds: (B, num_patches, hidden_size)
        # query_tokens: (B, num_query_tokens, hidden_size)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query_tokens, image_embeds, image_embeds
        )
        query_tokens = self.norm1(query_tokens + attn_out)
        
        # Self-attention
        attn_out, _ = self.self_attn(query_tokens, query_tokens, query_tokens)
        query_tokens = self.norm2(query_tokens + attn_out)
        
        # MLP
        mlp_out = self.mlp(query_tokens)
        query_tokens = self.norm3(query_tokens + mlp_out)
        
        return query_tokens  # (B, num_query_tokens, hidden_size)


class SimpleLLMProjection(nn.Module):
    """Project Q-Former output to LLM space."""
    def __init__(self, input_dim=768, output_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # x: (B, num_query_tokens, input_dim)
        return self.proj(x)  # (B, num_query_tokens, output_dim)


class SimpleLLM(nn.Module):
    """Simple LLM that generates logits."""
    def __init__(self, hidden_size=768, vocab_size=32000, num_tokens=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vocab_size),
        )
        
        self.num_tokens = num_tokens
    
    def forward(self, x):
        # x: (B, num_tokens, hidden_size)
        B, T, D = x.shape
        
        # Reshape to (B*T, D) for linear layer
        x = x.reshape(B*T, D)
        
        # Generate logits
        logits = self.decoder(x)  # (B*T, vocab_size)
        
        # Reshape back to (B, T, vocab_size)
        logits = logits.reshape(B, T, self.vocab_size)
        
        return logits  # (B, num_tokens, vocab_size)


def test_blip2_forward():
    """Test BLIP-2 forward pass with shapes."""
    
    print("\n" + "="*70)
    print("BLIP-2 FULL FORWARD PASS TEST")
    print("="*70)
    
    # Setup hyperparameters
    batch_size = 4
    img_channels = 3
    img_size = 224
    num_query_tokens = 32
    vision_hidden_dim = 768
    qformer_hidden_dim = 768
    llm_hidden_dim = 768
    llm_vocab_size = 32000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Step 1: Create a batch of dummy images
    print("\n[STEP 1] Create Input Batch")
    print("-" * 70)
    
    images = torch.randn(batch_size, img_channels, img_size, img_size, device=device)
    print(f"Input images shape: {images.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Channels: {img_channels}")
    print(f"  - Height: {img_size}")
    print(f"  - Width: {img_size}")
    
    # Step 2: Vision Encoder
    print("\n[STEP 2] Vision Encoder → Projection")
    print("-" * 70)
    
    vision_encoder = SimpleVisionEncoder().to(device)
    ln_vision = SimpleLayerNorm(vision_encoder.num_features).to(device)
    
    with torch.no_grad():
        # Vision encoder
        vision_output = vision_encoder(images)
        print(f"Vision encoder output shape: {vision_output.shape}")
        print(f"  - Batch size: {vision_output.shape[0]}")
        print(f"  - Number of patches: {vision_output.shape[1]}")
        print(f"  - Hidden dimension: {vision_output.shape[2]}")
        
        # Layer norm
        image_embeds = ln_vision(vision_output)
        print(f"After layer norm shape: {image_embeds.shape}")
    
    # Step 3: Q-Former
    print("\n[STEP 3] Q-Former Processing")
    print("-" * 70)
    
    qformer = SimpleQFormer(
        num_query_tokens=num_query_tokens,
        hidden_size=qformer_hidden_dim
    ).to(device)
    
    with torch.no_grad():
        # Expand query tokens for batch
        query_tokens = qformer.query_tokens.expand(batch_size, -1, -1)
        print(f"Query tokens shape: {query_tokens.shape}")
        print(f"  - Batch size: {query_tokens.shape[0]}")
        print(f"  - Number of query tokens: {query_tokens.shape[1]}")
        print(f"  - Hidden dimension: {query_tokens.shape[2]}")
        
        # Q-Former forward
        qformer_output = qformer(image_embeds, query_tokens)
        print(f"Q-Former output shape: {qformer_output.shape}")
        print(f"  - Batch size: {qformer_output.shape[0]}")
        print(f"  - Number of query tokens: {qformer_output.shape[1]}")
        print(f"  - Hidden dimension: {qformer_output.shape[2]}")
    
    # Step 4: Projection to LLM
    print("\n[STEP 4] Q-Former → LLM Projection")
    print("-" * 70)
    
    llm_projection = SimpleLLMProjection(
        input_dim=qformer_hidden_dim,
        output_dim=llm_hidden_dim
    ).to(device)
    
    with torch.no_grad():
        projected_output = llm_projection(qformer_output)
        print(f"Projected output shape: {projected_output.shape}")
        print(f"  - Batch size: {projected_output.shape[0]}")
        print(f"  - Number of tokens: {projected_output.shape[1]}")
        print(f"  - LLM hidden dimension: {projected_output.shape[2]}")
    
    # Step 5: LLM Forward Pass
    print("\n[STEP 5] LLM (Language Model) Forward Pass")
    print("-" * 70)
    
    llm = SimpleLLM(
        hidden_size=llm_hidden_dim,
        vocab_size=llm_vocab_size,
        num_tokens=num_query_tokens
    ).to(device)
    
    with torch.no_grad():
        logits = llm(projected_output)
        print(f"LLM logits shape: {logits.shape}")
        print(f"  - Batch size: {logits.shape[0]}")
        print(f"  - Sequence length: {logits.shape[1]}")
        print(f"  - Vocabulary size: {logits.shape[2]}")
    
    # Final Summary
    print("\n" + "="*70)
    print("FORWARD PASS SUMMARY")
    print("="*70)
    
    print(f"""
Vision Pipeline:
  Input:           {images.shape}
  → Vision Encoder → {vision_output.shape}
  → Layer Norm     → {image_embeds.shape}

Q-Former Pipeline:
  Query Tokens:    {query_tokens.shape}
  Vision Embeds:   {image_embeds.shape}
  → Q-Former       → {qformer_output.shape}

LLM Pipeline:
  Q-Former Out:    {qformer_output.shape}
  → Projection     → {projected_output.shape}
  → LLM            → {logits.shape}

Key Statistics:
  - Vision patches: {vision_output.shape[1]}
  - Query tokens: {num_query_tokens}
  - Vocabulary size: {llm_vocab_size}
  - Device: {device}
""")
    
    # Validation
    print("VALIDATION")
    print("-" * 70)
    
    checks = [
        ("Vision output dim", vision_output.shape[2] == vision_hidden_dim, f"{vision_output.shape[2]} == {vision_hidden_dim}"),
        ("Q-Former output dim", qformer_output.shape[2] == qformer_hidden_dim, f"{qformer_output.shape[2]} == {qformer_hidden_dim}"),
        ("Projected output dim", projected_output.shape[2] == llm_hidden_dim, f"{projected_output.shape[2]} == {llm_hidden_dim}"),
        ("LLM logits vocab dim", logits.shape[2] == llm_vocab_size, f"{logits.shape[2]} == {llm_vocab_size}"),
        ("Batch size preserved", logits.shape[0] == batch_size, f"{logits.shape[0]} == {batch_size}"),
        ("Query tokens preserved", logits.shape[1] == num_query_tokens, f"{logits.shape[1]} == {num_query_tokens}"),
        ("No NaN or Inf", not (torch.isnan(logits).any() or torch.isinf(logits).any()), "Valid tensors"),
    ]
    
    all_pass = True
    for check_name, result, detail in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name:30s} {detail}")
        if not result:
            all_pass = False
    
    print("\n" + "="*70)
    if all_pass:
        print("✓ ALL CHECKS PASSED - Forward pass successful!")
    else:
        print("✗ Some checks failed")
    print("="*70 + "\n")
    
    return all_pass


if __name__ == "__main__":
    success = test_blip2_forward()
    exit(0 if success else 1)
