#!/usr/bin/env python3
"""
Validation script for PVT v2 b2 integration with BLIP-2.
Tests:
1. Model initialization
2. Forward pass with shape validation
3. Parameter freezing
4. Integration with BLIP2 components
"""

import torch
import torch.nn as nn
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pvt_wrapper():
    """Test PVT v2 b2 wrapper standalone."""
    logger.info("=" * 60)
    logger.info("TEST 1: PVT v2 b2 Wrapper Initialization & Forward Pass")
    logger.info("=" * 60)
    
    try:
        from lavis.models.blip2_models.blip2 import PVTv2B2Wrapper
        
        # Initialize wrapper
        logger.info("1.1 Initializing PVT v2 b2 wrapper...")
        wrapper = PVTv2B2Wrapper(pretrained=True)
        
        logger.info(f"✓ Wrapper initialized")
        logger.info(f"  - Output dimension: {wrapper.num_features}")
        logger.info(f"  - Expected: 512")
        
        assert wrapper.num_features == 512, f"Expected num_features=512, got {wrapper.num_features}"
        
        # Test forward pass
        logger.info("\n1.2 Testing forward pass...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wrapper = wrapper.to(device)
        
        batch_size = 2
        img_size = 224
        x = torch.randn(batch_size, 3, img_size, img_size).to(device)
        
        logger.info(f"  Input shape: {x.shape}")
        
        with torch.no_grad():
            output = wrapper(x)
        
        logger.info(f"  Output shape: {output.shape}")
        
        # Validate output shape (B, N, C)
        expected_N = (img_size // 32) ** 2  # 7x7 = 49
        expected_shape = (batch_size, expected_N, 512)
        
        logger.info(f"  Expected shape: {expected_shape}")
        
        if output.shape == expected_shape:
            logger.info("✓ Output shape validation PASSED")
        else:
            logger.error(f"✗ Output shape mismatch: got {output.shape}, expected {expected_shape}")
            raise AssertionError(f"Shape mismatch: {output.shape} vs {expected_shape}")
        
        # Test parameter freezing
        logger.info("\n1.3 Validating parameter freezing...")
        frozen_count = 0
        trainable_count = 0
        for param in wrapper.parameters():
            if param.requires_grad:
                trainable_count += 1
            else:
                frozen_count += 1
        
        logger.info(f"  Frozen params: {frozen_count}")
        logger.info(f"  Trainable params: {trainable_count}")
        
        if trainable_count == 0:
            logger.info("✓ All parameters properly frozen")
        else:
            logger.warning(f"⚠️ Found {trainable_count} trainable parameters (expected 0)")
        
        logger.info("\n✓ TEST 1 PASSED\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST 1 FAILED: {str(e)}", exc_info=True)
        return False


def test_blip2_with_pvt():
    """Test PVT integration with BLIP2 components."""
    logger.info("=" * 60)
    logger.info("TEST 2: BLIP2 Integration with PVT v2 b2")
    logger.info("=" * 60)
    
    try:
        from lavis.models.blip2_models.blip2 import Blip2Base, LayerNorm
        
        # Create a minimal BLIP2-like model
        logger.info("2.1 Initializing Blip2Base with PVT v2 b2...")
        
        class TestBlip2(Blip2Base):
            def __init__(self):
                super().__init__()
                self.tokenizer = self.init_tokenizer()
                self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                    model_name="pvt_v2_b2",
                    img_size=224,
                    drop_path_rate=0,
                    use_grad_checkpoint=False,
                    precision="fp16"
                )
                
                # Initialize Q-former
                self.Qformer, self.query_tokens = self.init_Qformer(
                    num_query_token=32,
                    vision_width=self.visual_encoder.num_features,
                    cross_attention_freq=2
                )
                
                # Remove unnecessary Q-former components
                self.Qformer.cls = None
                self.Qformer.bert.embeddings.word_embeddings = None
                self.Qformer.bert.embeddings.position_embeddings = None
                for layer in self.Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None
        
        model = TestBlip2()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        logger.info(f"✓ BLIP2 model with PVT v2 b2 initialized")
        logger.info(f"  - Vision encoder type: {type(model.visual_encoder).__name__}")
        logger.info(f"  - Vision encoder output dim: {model.visual_encoder.num_features}")
        logger.info(f"  - Q-former hidden size: {model.Qformer.config.hidden_size}")
        
        # Test forward pass
        logger.info("\n2.2 Testing forward pass through vision encoder and Q-former...")
        
        batch_size = 2
        img_size = 224
        x = torch.randn(batch_size, 3, img_size, img_size).to(device)
        
        with torch.no_grad():
            # Get image embeddings
            image_embeds = model.ln_vision(model.visual_encoder(x))
            logger.info(f"  Image embeddings shape: {image_embeds.shape}")
            
            # Create attention mask
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            
            # Pass through Q-former
            query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            logger.info(f"  Query output shape: {query_output.last_hidden_state.shape}")
        
        # Validate shapes
        expected_embed_shape = (batch_size, (img_size // 32) ** 2, 512)  # (B, 49, 512)
        expected_query_shape = (batch_size, 32, model.Qformer.config.hidden_size)
        
        if image_embeds.shape == expected_embed_shape:
            logger.info(f"✓ Image embeddings shape correct: {image_embeds.shape}")
        else:
            logger.error(f"✗ Image embeddings shape mismatch: {image_embeds.shape} vs {expected_embed_shape}")
            raise AssertionError(f"Shape mismatch")
        
        if query_output.last_hidden_state.shape == expected_query_shape:
            logger.info(f"✓ Query output shape correct: {query_output.last_hidden_state.shape}")
        else:
            logger.error(f"✗ Query output shape mismatch: {query_output.last_hidden_state.shape} vs {expected_query_shape}")
            raise AssertionError(f"Shape mismatch")
        
        logger.info("\n✓ TEST 2 PASSED\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST 2 FAILED: {str(e)}", exc_info=True)
        return False


def test_model_loading():
    """Test loading models with PVT configuration."""
    logger.info("=" * 60)
    logger.info("TEST 3: Model Configuration Loading")
    logger.info("=" * 60)
    
    try:
        from lavis.models.blip2_models.blip2 import Blip2Base
        
        logger.info("3.1 Testing init_vision_encoder with pvt_v2_b2...")
        
        base_model = Blip2Base()
        visual_encoder, ln_vision = base_model.init_vision_encoder(
            model_name="pvt_v2_b2",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            precision="fp16"
        )
        
        logger.info(f"✓ Vision encoder initialized")
        logger.info(f"  - Type: {type(visual_encoder).__name__}")
        logger.info(f"  - Output dim (num_features): {visual_encoder.num_features}")
        logger.info(f"  - LayerNorm type: {type(ln_vision).__name__}")
        
        assert visual_encoder.num_features == 512, "Expected 512 output features"
        assert hasattr(visual_encoder, 'get_num_layer'), "Missing get_num_layer method"
        
        logger.info("\n✓ TEST 3 PASSED\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST 3 FAILED: {str(e)}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("PVT v2 b2 INTEGRATION TEST SUITE")
    logger.info("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("PVT Wrapper", test_pvt_wrapper()))
    results.append(("BLIP2 Integration", test_blip2_with_pvt()))
    results.append(("Model Loading", test_model_loading()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
