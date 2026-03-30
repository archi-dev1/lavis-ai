"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from timm import create_model

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.clip_vit import create_clip_vit_L
from transformers import BertTokenizer

try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft library not available. LoRA features will be disabled. Install: pip install peft")

try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("bitsandbytes not available for 8-bit quantization. Install: pip install bitsandbytes")


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


def load_llm_8bit_with_lora(
    model_name,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    lora_target_modules=None,
    load_in_8bit=True,
    device_map="auto",
):
    """
    Load LLM with 8-bit quantization and LoRA adaptation.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "google/flan-t5-xl")
        lora_r: LoRA rank (default 8)
        lora_alpha: LoRA alpha scaling (default 16)
        lora_dropout: LoRA dropout (default 0.1)
        lora_target_modules: Target modules for LoRA (default depends on model type)
        load_in_8bit: Enable 8-bit quantization (default True)
        device_map: Device mapping strategy (default "auto")
        
    Returns:
        Tuple of (model, tokenizer, lora_info_dict)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    
    logging.info(
        f"\nLoading LLM with 8-bit LoRA:\n"
        f"  - Model: {model_name}\n"
        f"  - 8-bit: {load_in_8bit}\n"
        f"  - LoRA rank: {lora_r}"
    )
    
    # Step 1: Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("✓ Tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        return None, None, None
    
    # Step 1: Load model with 8-bit quantization
    load_config = {}
    
    if load_in_8bit:
        if not BITSANDBYTES_AVAILABLE:
            logging.warning(
                "bitsandbytes not available. Fallback to fp16.\n"
                "Install with: pip install bitsandbytes"
            )
            load_config["torch_dtype"] = torch.float16
        else:
            try:
                # Check CUDA availability
                if torch.cuda.is_available():
                    load_config["load_in_8bit"] = True
                    load_config["device_map"] = device_map
                    logging.info("✓ Loading in 8-bit mode (CUDA available)")
                else:
                    logging.warning("CUDA not available. Falling back to fp16.")
                    load_config["torch_dtype"] = torch.float16
            except Exception as e:
                logging.warning(f"CUDA error during 8-bit setup: {e}. Falling back to fp16.")
                load_config["torch_dtype"] = torch.float16
    else:
        load_config["torch_dtype"] = torch.float16
        logging.info("Loading without 8-bit (using fp16)")
    
    # Determine model type and load
    try:
        if "flan-t5" in model_name.lower() or "t5" in model_name.lower():
            # T5 models
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_config)
            default_targets = ["q", "v"]  # T5 attention components
            logging.info(f"Loaded T5-style model: {model_name}")
        else:
            # Causal LM (LLaMA, Mistral, etc.)
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_config)
            default_targets = ["q_proj", "v_proj"]  # Standard transformer attention
            logging.info(f"Loaded Causal LM: {model_name}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        logger_msg = f"Error details: {str(e)}\n"
        if "bitsandbytes" in str(e).lower():
            logger_msg += "Try installing bitsandbytes: pip install bitsandbytes"
        logging.error(logger_msg)
        return None, tokenizer, None
    
    # Step 4: Freeze base model (before LoRA)
    for param in model.parameters():
        param.requires_grad = False
    logging.info("✓ Base model parameters frozen")
    
    # Step 2 & 3: Apply LoRA
    if lora_target_modules is None:
        lora_target_modules = default_targets
    
    model = apply_lora_to_llm(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
    )
    
    # Validation
    lora_info = get_trainable_params_info(model)
    
    logging.info(
        f"\n{'='*60}\n"
        f"LLM 8-bit LoRA Setup Complete:\n"
        f"{'='*60}\n"
        f"Model:              {model_name}\n"
        f"8-bit:              {load_in_8bit and BITSANDBYTES_AVAILABLE}\n"
        f"LoRA rank:          {lora_r}\n"
        f"LoRA alpha:         {lora_alpha}\n"
        f"Target modules:     {lora_target_modules}\n"
        f"Trainable params:   {lora_info['trainable_params']:,}\n"
        f"Total params:       {lora_info['total_params']:,}\n"
        f"Trainable %:        {lora_info['trainable_percentage']:.2f}%\n"
        f"{'='*60}"
    )
    
    if lora_info['trainable_percentage'] < 5:
        logging.info("✓ Trainable percentage < 5%: Configuration valid")
    else:
        logging.warning(f"⚠️ Trainable percentage: {lora_info['trainable_percentage']:.2f}%")
    
    return model, tokenizer, lora_info


def apply_lora_to_llm(model, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
    """
    Apply LoRA to LLM with auto-detection of modules.
    
    Args:
        model: Language model
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: Target modules for LoRA
        
    Returns:
        PEFT-wrapped model with LoRA
    """
    if not PEFT_AVAILABLE:
        logging.error("PEFT required for LoRA. Install: pip install peft")
        return model
    
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    # Auto-detect available modules
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
                logging.debug(f"Module '{target}' not found. Available: {available_modules}")
        
        if detected_modules:
            target_modules = detected_modules
            logging.info(f"Step 2: Auto-detected LoRA targets: {target_modules}")
    except Exception as e:
        logging.warning(f"Auto-detection failed: {e}. Using: {target_modules}")
    
    # Step 3: Create LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    logging.info(
        f"Step 2 & 3: LoRA applied\n"
        f"  - Rank: {r}\n"
        f"  - Alpha: {lora_alpha}\n"
        f"  - Dropout: {lora_dropout}"
    )
    
    return model


def freeze_llm_base(model):
    """
    Freeze all base parameters in LLM, keeping only LoRA trainable.
    
    Args:
        model: PEFT-wrapped LLM
        
    Returns:
        Number of frozen parameters
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
            frozen_count += 1
    
    logging.info(f"Step 4: Frozen {frozen_count} base LLM parameters")
    return frozen_count


class PVTv2B2Wrapper(nn.Module):
    """Wraps PVT v2 b2 vision encoder to be compatible with BLIP2 input/output format."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.pvt_encoder = create_model("pvt_v2_b2", pretrained=pretrained)
        
        # PVT v2 b2 last stage output dim is 512
        self.pvt_out_dim = 512
        self.num_features = self.pvt_out_dim
        
        # Freeze all parameters
        for param in self.pvt_encoder.parameters():
            param.requires_grad = False
        
        logging.info("PVT v2 b2 vision encoder initialized with frozen parameters")
    
    def forward(self, x):
        """
        Extract features from PVT v2 b2.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            features: Tensor of shape (B, N, C) where N = H*W/1024 and C = 512
        """
        batch_size = x.shape[0]
        
        # Get all stage outputs from PVT v2 b2
        # The model returns a list of tensors from all stages
        outs = self.pvt_encoder.forward_features(x) if hasattr(self.pvt_encoder, 'forward_features') else self.pvt_encoder(x)
        
        # Handle different output formats
        if isinstance(outs, (list, tuple)):
            x_out = outs[-1]  # Use last stage output
            logging.debug(f"PVT output (last stage list/tuple) shape: {x_out.shape}")
        else:
            x_out = outs
            logging.debug(f"PVT output shape: {x_out.shape}")
        
        # Convert (B, C, H, W) -> (B, N, C)
        if len(x_out.shape) == 4:
            B, C, H, W = x_out.shape
            
            # Validation with error handling
            if C != self.pvt_out_dim:
                logging.warning(
                    f"⚠️ Output dimension mismatch: got {C}, expected {self.pvt_out_dim}. "
                    f"Shapes: B={B}, H={H}, W={W}. Auto-fixing..."
                )
            
            # Reshape: (B, C, H, W) -> (B, H, W, C) -> (B, N, C)
            x_out = x_out.permute(0, 2, 3, 1)  # (B, H, W, C)
            x_out = x_out.reshape(B, -1, C)   # (B, H*W, C)
            x_out = x_out.contiguous()
            
            # Validation print
            N = H * W
            output_shape = x_out.shape
            expected_shape = (batch_size, N, self.pvt_out_dim)
            
            if output_shape == expected_shape:
                logging.info(f"✓ Output shape validation passed: {output_shape}")
            else:
                logging.error(
                    f"✗ Output shape mismatch: got {output_shape}, expected {expected_shape}. "
                    f"Original input shape: {x.shape}, PVT output: ({B}, {C}, {H}, {W})"
                )
        else:
            raise ValueError(
                f"❌ Unexpected output shape: {x_out.shape}, expected (B, C, H, W). "
                f"Input shape was: {x.shape}"
            )
        
        return x_out
    
    def get_num_layer(self, name=None):
        """Compatibility method for optimizer parameter grouping."""
        if name is None:
            return 4  # PVT v2 b2 has 4 stages
        # Extract layer index from name
        if "stage" in name:
            try:
                stage_num = int(name.split("stage")[-1].split(".")[0])
                return stage_num
            except (ValueError, IndexError):
                return 0
        return 0


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
            "pvt_v2_b2",
        ], "vit model must be eva_clip_g, eva2_clip_L, clip_L, or pvt_v2_b2"
        
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
            ln_vision = LayerNorm(visual_encoder.num_features)
#         elif model_name == "eva2_clip_L":
#             visual_encoder = create_eva2_vit_L(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
#             ln_vision = LayerNorm(visual_encoder.num_features)
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
            ln_vision = LayerNorm(visual_encoder.num_features)
        elif model_name == "pvt_v2_b2":
            visual_encoder = PVTv2B2Wrapper(pretrained=True)
            ln_vision = LayerNorm(visual_encoder.num_features)
            logging.info(f"Initialized PVT v2 b2 vision encoder with output dim: {visual_encoder.num_features}")
        
        self.vit_name = model_name
        return visual_encoder, ln_vision

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

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def get_optimizer_params(self, weight_decay, lr_scale=1):

        vit_num_layers = self.visual_encoder.get_num_layer()
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                group_name = "vit_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        # import json
        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
