#!/usr/bin/env python3
"""
Full BLIP-2 + PVT v2 b2 + QFormer LoRA training on MS COCO (Karpathy splits).

Designed to run on a college server via SSH.

Usage:
    # 1. SSH into your server
    # 2. Activate the environment
    source activate blip_source_env        # or: conda activate blip_source_env

    # 3. Run the script — it will prompt for Kaggle credentials
    python train_coco_server.py

    # Or pass Kaggle credentials via environment variables:
    KAGGLE_USERNAME=your_user KAGGLE_KEY=your_key python train_coco_server.py

    # To resume from a checkpoint:
    python train_coco_server.py --resume /path/to/checkpoint.pt

    # To run inference only on a saved checkpoint:
    python train_coco_server.py --inference-only --resume /path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
import warnings
from getpass import getpass
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# User-editable settings
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE          = 224
BATCH_SIZE_TRAIN    = 16          # per-GPU batch size (reduce if OOM)
BATCH_SIZE_EVAL     = 32
GRAD_ACCUM_STEPS    = 2           # effective batch = BATCH_SIZE_TRAIN * GRAD_ACCUM_STEPS
NUM_WORKERS         = 4
EPOCHS              = 5
LEARNING_RATE       = 1e-4
MIN_LR              = 1e-6
WARMUP_STEPS        = 500
WEIGHT_DECAY        = 0.05
SEED                = 42
MAX_TXT_LEN         = 40
PROMPT              = "a photo of "

# Generation settings
NUM_BEAMS           = 5
MAX_GEN_LENGTH      = 30
MIN_GEN_LENGTH      = 5
NO_REPEAT_NGRAM     = 3

# Model settings
VIT_MODEL           = "pvt_v2_b2"
T5_MODEL            = "google/flan-t5-small"   # use flan-t5-base or flan-t5-xl if you have VRAM
NUM_QUERY_TOKENS    = 16
LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.1

# Paths — adjust if your server layout differs
SCRIPT_DIR          = Path(__file__).resolve().parent
DATA_ROOT           = SCRIPT_DIR / "data" / "coco"
IMAGES_DIR          = DATA_ROOT / "images"
ANNOTATIONS_DIR     = DATA_ROOT / "annotations"
OUTPUT_DIR          = SCRIPT_DIR / "output" / "blip2_coco_pvt_lora"

# Kaggle dataset identifiers for COCO images
KAGGLE_COCO_TRAIN   = "nikhil7280/coco-image-caption"  # train2014 + val2014 images + annotations
# Karpathy split JSON URLs (hosted by Salesforce)
KARPATHY_URLS = {
    "train": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json",
    "val":   "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json",
    "test":  "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json",
}

LOG_EVERY = 50   # print loss every N steps

# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
# 0. Seed
# ═════════════════════════════════════════════════════════════
def set_seed(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ═════════════════════════════════════════════════════════════
# 1. Kaggle credentials
# ═════════════════════════════════════════════════════════════
def setup_kaggle_credentials():
    """Prompt for Kaggle username & key if not already configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        kaggle_json.chmod(0o600)
        logger.info(f"Found existing Kaggle credentials at {kaggle_json}")
        return

    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()

    if not username or not key:
        print("\n╔══════════════════════════════════════════╗")
        print("║   Kaggle credentials required            ║")
        print("║   (get them from kaggle.com/settings)    ║")
        print("╚══════════════════════════════════════════╝")
        username = input("Kaggle username: ").strip()
        key = getpass("Kaggle key: ").strip()

    if not username or not key:
        raise RuntimeError(
            "Kaggle credentials are required to download COCO images.\n"
            "Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or enter them at the prompt."
        )

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json.write_text(json.dumps({"username": username, "key": key}))
    kaggle_json.chmod(0o600)
    logger.info(f"Saved Kaggle credentials to {kaggle_json}")


# ═════════════════════════════════════════════════════════════
# 2. Download COCO images + Karpathy annotations
# ═════════════════════════════════════════════════════════════
def download_coco_images():
    """Download COCO images from Kaggle if not present."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Check if images already exist
    jpg_count = len(list(IMAGES_DIR.rglob("*.jpg")))
    if jpg_count >= 1000:
        logger.info(f"COCO images already present ({jpg_count} .jpg files). Skipping download.")
        return

    logger.info("Downloading COCO images from Kaggle (this may take a while)...")
    cmd = [
        "kaggle", "datasets", "download", "-d", KAGGLE_COCO_TRAIN,
        "-p", str(DATA_ROOT), "--unzip",
    ]
    subprocess.run(cmd, check=True)

    # The Kaggle dataset may unpack into a subfolder — find the images
    # and flatten into IMAGES_DIR if needed.
    _organise_images()


def _organise_images():
    """
    After Kaggle unzip, image folders could be nested.
    We need all images accessible as:
        IMAGES_DIR/train2014/COCO_train2014_XXXX.jpg
        IMAGES_DIR/val2014/COCO_val2014_XXXX.jpg
    """
    # Look for train2014/val2014 folders anywhere under DATA_ROOT
    for split_name in ("train2014", "val2014"):
        target = IMAGES_DIR / split_name
        if target.is_dir() and len(list(target.glob("*.jpg"))) > 100:
            continue  # already good

        # Search recursively
        for candidate in DATA_ROOT.rglob(split_name):
            if candidate.is_dir() and candidate != target:
                jpgs = list(candidate.glob("*.jpg"))
                if len(jpgs) > 100:
                    logger.info(f"Moving {candidate} -> {target}")
                    if target.exists():
                        import shutil
                        shutil.rmtree(target)
                    candidate.rename(target)
                    break

    # Verify
    for split_name in ("train2014", "val2014"):
        d = IMAGES_DIR / split_name
        if d.is_dir():
            count = len(list(d.glob("*.jpg")))
            logger.info(f"  {split_name}: {count} images")
        else:
            # Sometimes images are directly in IMAGES_DIR or a flat folder
            logger.warning(f"  {split_name} not found as subfolder. Checking flat layout...")
            flat_count = len(list(IMAGES_DIR.glob(f"COCO_{split_name}_*.jpg")))
            if flat_count > 0:
                logger.info(f"  Found {flat_count} {split_name} images in flat layout — creating subfolder")
                d.mkdir(parents=True, exist_ok=True)
                for img in IMAGES_DIR.glob(f"COCO_{split_name}_*.jpg"):
                    img.rename(d / img.name)


def download_karpathy_annotations():
    """Download Karpathy split JSON files from Salesforce."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request

    for split_name, url in KARPATHY_URLS.items():
        dest = ANNOTATIONS_DIR / f"coco_karpathy_{split_name}.json"
        if dest.exists():
            logger.info(f"  {dest.name} already exists — skipping")
            continue
        logger.info(f"  Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, str(dest))
        logger.info(f"  Saved to {dest}")


# ═════════════════════════════════════════════════════════════
# 3. Dataset & DataLoader
# ═════════════════════════════════════════════════════════════
class COCOKarpathyCaptionDataset:
    """
    Lightweight COCO Karpathy dataset that reads the Salesforce Karpathy JSON
    and returns image tensors + captions suitable for Blip2T5.forward().
    """

    def __init__(self, ann_path, images_root, split, transform, prompt,
                 max_captions_per_image=5):
        self.images_root = Path(images_root)
        self.transform = transform
        self.prompt = prompt
        self.samples = []

        with open(ann_path, "r") as f:
            data = json.load(f)

        # The Salesforce Karpathy JSONs are a list of dicts, each with keys:
        #   image, caption  (for train)
        #   image, caption  (for val/test — may have list or string)
        # Format varies slightly: some are flat lists, some have "annotations" key.
        if isinstance(data, list):
            annotations = data
        elif "annotations" in data:
            annotations = data["annotations"]
        elif "images" in data:
            # Full Karpathy format with images->sentences
            annotations = self._parse_karpathy_full(data, split, max_captions_per_image)
        else:
            annotations = data

        for ann in annotations:
            image_path = ann.get("image", "")
            caption = ann.get("caption", "")
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            caption = str(caption).strip()
            if image_path and caption:
                self.samples.append({
                    "image": image_path,
                    "caption": caption.lower().strip(),
                })

        logger.info(f"[{split.upper()}] Loaded {len(self.samples)} image-caption pairs")
        if self.samples:
            logger.info(f"  Sample: {self.samples[0]['caption'][:80]}...")

    @staticmethod
    def _parse_karpathy_full(data, split, max_captions):
        """Parse the full Karpathy JSON (images -> sentences structure)."""
        annotations = []
        for img in data["images"]:
            if img.get("split", "") != split:
                continue
            filename = img.get("filename", "")
            for i, sent in enumerate(img.get("sentences", [])):
                if i >= max_captions:
                    break
                annotations.append({
                    "image": filename,
                    "caption": sent.get("raw", ""),
                })
        return annotations

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image as PILImage

        max_retries = 5
        for attempt in range(max_retries):
            try:
                ann = self.samples[index]
                img_path = self.images_root / ann["image"]

                # If the file doesn't exist at top level, try inside train2014/val2014
                if not img_path.exists():
                    for sub in ("train2014", "val2014"):
                        alt = self.images_root / sub / Path(ann["image"]).name
                        if alt.exists():
                            img_path = alt
                            break

                with PILImage.open(img_path) as img:
                    img = img.convert("RGB")
                    img_tensor = self.transform(img)

                return {
                    "image": img_tensor,
                    "text_input": self.prompt,
                    "text_output": ann["caption"],
                }
            except Exception:
                index = random.randint(0, len(self.samples) - 1)

        raise RuntimeError(f"Failed to load sample after {max_retries} retries")


class COCOKarpathyEvalDataset_:
    """Eval dataset — returns image + all reference captions for metrics."""

    def __init__(self, ann_path, images_root, split, transform, prompt):
        self.images_root = Path(images_root)
        self.transform = transform
        self.prompt = prompt
        self.samples = []

        with open(ann_path, "r") as f:
            data = json.load(f)

        # Group captions by image
        image_captions: dict[str, list[str]] = {}

        if isinstance(data, list):
            entries = data
        elif "annotations" in data:
            entries = data["annotations"]
        elif "images" in data:
            entries = None
            for img in data["images"]:
                if img.get("split", "") != split:
                    continue
                fname = img.get("filename", "")
                caps = [s["raw"].lower().strip() for s in img.get("sentences", [])]
                if fname and caps:
                    image_captions[fname] = caps
        else:
            entries = data if isinstance(data, list) else []

        if entries is not None:
            for ann in entries:
                img = ann.get("image", "")
                cap = ann.get("caption", "")
                if isinstance(cap, list):
                    cap = cap[0] if cap else ""
                cap = str(cap).lower().strip()
                if img and cap:
                    image_captions.setdefault(img, []).append(cap)

        for image_path, captions in image_captions.items():
            self.samples.append({
                "image": image_path,
                "captions": captions,
            })

        logger.info(f"[{split.upper()} EVAL] Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image as PILImage

        max_retries = 5
        for attempt in range(max_retries):
            try:
                ann = self.samples[index]
                img_path = self.images_root / ann["image"]

                if not img_path.exists():
                    for sub in ("train2014", "val2014"):
                        alt = self.images_root / sub / Path(ann["image"]).name
                        if alt.exists():
                            img_path = alt
                            break

                with PILImage.open(img_path) as img:
                    img = img.convert("RGB")
                    img_tensor = self.transform(img)

                return {
                    "image": img_tensor,
                    "captions": ann["captions"],
                    "image_path": ann["image"],
                }
            except Exception:
                index = random.randint(0, len(self.samples) - 1)

        raise RuntimeError(f"Failed to load eval sample after {max_retries} retries")


def build_transforms(image_size):
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def collate_fn(batch):
    import torch

    images = torch.stack([b["image"] for b in batch])
    text_input = [b["text_input"] for b in batch]
    text_output = [b["text_output"] for b in batch]
    return {"image": images, "text_input": text_input, "text_output": text_output}


def eval_collate_fn(batch):
    import torch

    images = torch.stack([b["image"] for b in batch])
    captions = [b["captions"] for b in batch]
    paths = [b["image_path"] for b in batch]
    return {"image": images, "captions": captions, "image_path": paths}


# ═════════════════════════════════════════════════════════════
# 4. Model
# ═════════════════════════════════════════════════════════════
def build_model(device):
    logger.info("Building BLIP-2  (PVT v2 b2 + QFormer LoRA + FlanT5)...")

    # Make sure repo root is on path
    repo_root = str(SCRIPT_DIR)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from lavis.models.blip2_models.blip2 import (
        apply_lora_to_qformer,
        freeze_qformer_base,
        get_trainable_params_info,
    )
    from lavis.models.blip2_models.blip2_t5 import Blip2T5

    model = Blip2T5(
        vit_model=VIT_MODEL,
        img_size=IMAGE_SIZE,
        num_query_token=NUM_QUERY_TOKENS,
        t5_model=T5_MODEL,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        prompt=PROMPT,
        max_txt_len=MAX_TXT_LEN,
    )

    # Apply LoRA to QFormer
    model.Qformer = apply_lora_to_qformer(
        model.Qformer,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "key", "value"],
    )
    freeze_qformer_base(model.Qformer)

    # Freeze vision encoder
    for param in model.visual_encoder.parameters():
        param.requires_grad = False
    for param in model.ln_vision.parameters():
        param.requires_grad = False

    # Freeze T5, keep projection trainable
    for param in model.t5_model.parameters():
        param.requires_grad = False
        param.data = param.data.float()
    for param in model.t5_proj.parameters():
        param.requires_grad = True

    model = model.to(device)
    info = get_trainable_params_info(model)
    logger.info(
        f"Trainable: {info['trainable_params']:,} / {info['total_params']:,} "
        f"({info['trainable_percentage']:.2f}%)"
    )
    return model


# ═════════════════════════════════════════════════════════════
# 5. Learning-rate scheduler (linear warmup + cosine decay)
# ═════════════════════════════════════════════════════════════
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    import math

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ═════════════════════════════════════════════════════════════
# 6. Training loop
# ═════════════════════════════════════════════════════════════
def train(model, train_loader, val_loader, device, args):
    import torch

    logger.info("Starting training...")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, WARMUP_STEPS, total_steps, min_lr_ratio=MIN_LR / LEARNING_RATE
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # Resume from checkpoint
    if args.resume and Path(args.resume).exists():
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "global_step" in ckpt:
            global_step = ckpt["global_step"]
        if "best_val_loss" in ckpt:
            best_val_loss = ckpt["best_val_loss"]
        logger.info(f"Resumed at epoch {start_epoch}, step {global_step}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader, 1):
            batch["image"] = batch["image"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                outputs = model(batch)
                loss = outputs["loss"] / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if batch_idx % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            loss_val = loss.item() * GRAD_ACCUM_STEPS
            epoch_loss += loss_val
            epoch_steps += 1

            if batch_idx % LOG_EVERY == 0:
                avg = epoch_loss / epoch_steps
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch+1}/{EPOCHS} | Step {batch_idx}/{len(train_loader)} | "
                    f"Loss {loss_val:.4f} | Avg {avg:.4f} | LR {lr:.2e} | {elapsed:.0f}s"
                )

        # Flush remaining grads
        if batch_idx % GRAD_ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")

        # Validation
        val_loss = validate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} val loss:   {val_loss:.4f}")

        # Save checkpoint
        ckpt_path = OUTPUT_DIR / f"checkpoint_epoch{epoch+1}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "config": {
                "vit_model": VIT_MODEL, "t5_model": T5_MODEL,
                "image_size": IMAGE_SIZE, "num_query_tokens": NUM_QUERY_TOKENS,
                "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            },
        }, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = OUTPUT_DIR / "best_model.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "config": {
                    "vit_model": VIT_MODEL, "t5_model": T5_MODEL,
                    "image_size": IMAGE_SIZE, "num_query_tokens": NUM_QUERY_TOKENS,
                    "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                },
            }, best_path)
            logger.info(f"New best model saved: val_loss={best_val_loss:.4f}")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def validate(model, val_loader, device):
    import torch

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch["image"] = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                outputs = model(batch)
            total_loss += outputs["loss"].item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


# ═════════════════════════════════════════════════════════════
# 7. Inference — caption generation
# ═════════════════════════════════════════════════════════════
def run_inference(model, eval_loader, device, split_name="test", max_images=50):
    """Generate captions for images and display results."""
    import torch

    logger.info(f"\n{'='*60}")
    logger.info(f"Generating captions on {split_name} set (up to {max_images} images)")
    logger.info(f"{'='*60}")

    model.eval()
    results = []
    count = 0

    with torch.no_grad():
        for batch in eval_loader:
            if count >= max_images:
                break

            images = batch["image"].to(device, non_blocking=True)
            ref_captions = batch["captions"]
            image_paths = batch["image_path"]

            generated = model.generate(
                {"image": images, "prompt": PROMPT},
                use_nucleus_sampling=False,
                num_beams=NUM_BEAMS,
                max_length=MAX_GEN_LENGTH,
                min_length=MIN_GEN_LENGTH,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
            )

            for i, (gen_cap, refs, img_path) in enumerate(
                zip(generated, ref_captions, image_paths)
            ):
                if count >= max_images:
                    break
                count += 1
                gen_cap = gen_cap.strip()
                results.append({
                    "image": img_path,
                    "generated": gen_cap,
                    "references": refs,
                })

                print(f"\n[{count}] Image: {img_path}")
                print(f"    Generated:  {gen_cap}")
                if refs:
                    print(f"    Reference:  {refs[0]}")

    # Save results to JSON
    results_path = OUTPUT_DIR / f"captions_{split_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved {len(results)} captions to {results_path}")

    model.train()
    return results


# ═════════════════════════════════════════════════════════════
# 8. Main
# ═════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="BLIP-2 COCO Training (PVT + LoRA)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--inference-only", action="store_true", help="Skip training, run inference only")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--t5-model", type=str, default=None, help="Override T5 model name")
    parser.add_argument("--max-inference", type=int, default=50, help="Max images for inference")
    return parser.parse_args()


def main():
    args = parse_args()

    # Allow CLI overrides
    global BATCH_SIZE_TRAIN, EPOCHS, LEARNING_RATE, T5_MODEL
    if args.batch_size:
        BATCH_SIZE_TRAIN = args.batch_size
    if args.epochs:
        EPOCHS = args.epochs
    if args.lr:
        LEARNING_RATE = args.lr
    if args.t5_model:
        T5_MODEL = args.t5_model

    set_seed(SEED)

    import torch
    from torch.utils.data import DataLoader

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    else:
        logger.warning("No GPU detected — training will be very slow!")

    # ── Step 1: Kaggle credentials ──
    logger.info("\n[Step 1/6] Setting up Kaggle credentials...")
    setup_kaggle_credentials()

    # ── Step 2: Download data ──
    logger.info("\n[Step 2/6] Downloading COCO images...")
    download_coco_images()

    logger.info("\n[Step 3/6] Downloading Karpathy annotations...")
    download_karpathy_annotations()

    # ── Step 3: Build datasets ──
    logger.info("\n[Step 4/6] Building datasets...")
    train_transform, eval_transform = build_transforms(IMAGE_SIZE)

    train_ann = ANNOTATIONS_DIR / "coco_karpathy_train.json"
    val_ann = ANNOTATIONS_DIR / "coco_karpathy_val.json"
    test_ann = ANNOTATIONS_DIR / "coco_karpathy_test.json"

    train_dataset = COCOKarpathyCaptionDataset(
        ann_path=str(train_ann), images_root=str(IMAGES_DIR),
        split="train", transform=train_transform, prompt=PROMPT,
    )
    val_dataset_train_format = COCOKarpathyCaptionDataset(
        ann_path=str(val_ann), images_root=str(IMAGES_DIR),
        split="val", transform=eval_transform, prompt=PROMPT,
    )
    val_dataset_eval = COCOKarpathyEvalDataset_(
        ann_path=str(val_ann), images_root=str(IMAGES_DIR),
        split="val", transform=eval_transform, prompt=PROMPT,
    )
    test_dataset_eval = COCOKarpathyEvalDataset_(
        ann_path=str(test_ann), images_root=str(IMAGES_DIR),
        split="test", transform=eval_transform, prompt=PROMPT,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset_train_format, batch_size=BATCH_SIZE_EVAL, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn,
    )
    val_eval_loader = DataLoader(
        val_dataset_eval, batch_size=BATCH_SIZE_EVAL, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=eval_collate_fn,
    )
    test_eval_loader = DataLoader(
        test_dataset_eval, batch_size=BATCH_SIZE_EVAL, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=eval_collate_fn,
    )

    logger.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset_train_format)} | "
                f"Val eval: {len(val_dataset_eval)} | Test eval: {len(test_dataset_eval)}")

    # ── Step 4: Build model ──
    logger.info("\n[Step 5/6] Building model...")
    model = build_model(device)

    # ── Step 5: Train ──
    if not args.inference_only:
        logger.info("\n[Step 6/6] Training...")
        logger.info(f"Config: epochs={EPOCHS}, bs={BATCH_SIZE_TRAIN}, grad_accum={GRAD_ACCUM_STEPS}, "
                     f"lr={LEARNING_RATE}, t5={T5_MODEL}")
        train(model, train_loader, val_loader, device, args)
    else:
        if args.resume:
            logger.info(f"Loading checkpoint for inference: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)

    # ── Step 6: Inference ──
    logger.info("\nRunning inference on validation set...")
    run_inference(model, val_eval_loader, device, split_name="val", max_images=args.max_inference)

    logger.info("\nRunning inference on test set...")
    run_inference(model, test_eval_loader, device, split_name="test", max_images=args.max_inference)

    logger.info(f"\n{'='*60}")
    logger.info("DONE!")
    logger.info(f"Checkpoints saved in: {OUTPUT_DIR}")
    logger.info(f"Generated captions in: {OUTPUT_DIR / 'captions_*.json'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
