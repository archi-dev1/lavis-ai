#!/usr/bin/env python3
"""
BLIP-2 + PVT v2 b2 + QFormer LoRA inference on MS COCO (Karpathy splits).

Loads a checkpoint from training and generates captions on val/test sets.

Usage:
    # Run inference on best model checkpoint
    python infer_coco_server.py --checkpoint output/blip2_coco_pvt_lora/best_model.pt

    # Run on a specific epoch checkpoint
    python infer_coco_server.py --checkpoint output/blip2_coco_pvt_lora/checkpoint_epoch5.pt

    # Limit number of images
    python infer_coco_server.py --checkpoint output/blip2_coco_pvt_lora/best_model.pt --max-images 100

    # Run only on test set
    python infer_coco_server.py --checkpoint output/blip2_coco_pvt_lora/best_model.pt --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import warnings
from getpass import getpass
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Settings (must match training script)
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE          = 224
BATCH_SIZE_EVAL     = 32
NUM_WORKERS         = 4
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
T5_MODEL            = "google/flan-t5-small"
NUM_QUERY_TOKENS    = 16
LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.1

# Paths
SCRIPT_DIR          = Path(__file__).resolve().parent
DATA_ROOT           = SCRIPT_DIR / "data" / "coco"
IMAGES_DIR          = DATA_ROOT / "images"
ANNOTATIONS_DIR     = DATA_ROOT / "annotations"
OUTPUT_DIR          = SCRIPT_DIR / "output" / "blip2_coco_pvt_lora"

# Kaggle dataset identifiers for COCO images
KAGGLE_COCO_TRAIN   = "nikhil7280/coco-image-caption"
# Karpathy split JSON URLs (hosted by Salesforce)
KARPATHY_URLS = {
    "train": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json",
    "val":   "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json",
    "test":  "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json",
}

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
    _organise_images()


def _organise_images():
    for split_name in ("train2014", "val2014"):
        target = IMAGES_DIR / split_name
        if target.is_dir() and len(list(target.glob("*.jpg"))) > 100:
            continue

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

    for split_name in ("train2014", "val2014"):
        d = IMAGES_DIR / split_name
        if d.is_dir():
            count = len(list(d.glob("*.jpg")))
            logger.info(f"  {split_name}: {count} images")
        else:
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
# 3. Eval Dataset & DataLoader
# ═════════════════════════════════════════════════════════════
class COCOKarpathyEvalDataset:
    """Eval dataset — returns image + all reference captions for metrics."""

    def __init__(self, ann_path, images_root, split, transform, prompt):
        self.images_root = Path(images_root)
        self.transform = transform
        self.prompt = prompt
        self.samples = []

        with open(ann_path, "r") as f:
            data = json.load(f)

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


def build_eval_transform(image_size):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


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

    model.Qformer = apply_lora_to_qformer(
        model.Qformer,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "key", "value"],
    )
    freeze_qformer_base(model.Qformer)

    for param in model.visual_encoder.parameters():
        param.requires_grad = False
    for param in model.ln_vision.parameters():
        param.requires_grad = False
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
# 5. Inference — caption generation
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

    results_path = OUTPUT_DIR / f"captions_{split_name}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved {len(results)} captions to {results_path}")

    return results


# ═════════════════════════════════════════════════════════════
# 6. Main
# ═════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="BLIP-2 COCO Inference (PVT + LoRA)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--split", type=str, default="both", choices=["val", "test", "both"],
                        help="Which split to run inference on (default: both)")
    parser.add_argument("--max-images", type=int, default=50,
                        help="Max images to generate captions for per split")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override eval batch size")
    parser.add_argument("--t5-model", type=str, default=None,
                        help="Override T5 model name")
    return parser.parse_args()


def main():
    args = parse_args()

    global BATCH_SIZE_EVAL, T5_MODEL
    if args.batch_size:
        BATCH_SIZE_EVAL = args.batch_size
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

    # ── Step 1: Ensure data is available ──
    logger.info("\n[Step 1/4] Checking Kaggle credentials...")
    setup_kaggle_credentials()

    logger.info("\n[Step 2/4] Ensuring COCO data is available...")
    download_coco_images()
    download_karpathy_annotations()

    # ── Step 2: Build eval datasets ──
    logger.info("\n[Step 3/4] Building eval datasets...")
    eval_transform = build_eval_transform(IMAGE_SIZE)

    loaders = {}
    if args.split in ("val", "both"):
        val_ann = ANNOTATIONS_DIR / "coco_karpathy_val.json"
        val_dataset = COCOKarpathyEvalDataset(
            ann_path=str(val_ann), images_root=str(IMAGES_DIR),
            split="val", transform=eval_transform, prompt=PROMPT,
        )
        loaders["val"] = DataLoader(
            val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=eval_collate_fn,
        )

    if args.split in ("test", "both"):
        test_ann = ANNOTATIONS_DIR / "coco_karpathy_test.json"
        test_dataset = COCOKarpathyEvalDataset(
            ann_path=str(test_ann), images_root=str(IMAGES_DIR),
            split="test", transform=eval_transform, prompt=PROMPT,
        )
        loaders["test"] = DataLoader(
            test_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=eval_collate_fn,
        )

    # ── Step 3: Build model & load checkpoint ──
    logger.info("\n[Step 4/4] Building model and loading checkpoint...")
    model = build_model(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    if "epoch" in ckpt:
        logger.info(f"  Trained for {ckpt['epoch'] + 1} epochs")
    if "best_val_loss" in ckpt:
        logger.info(f"  Best val loss: {ckpt['best_val_loss']:.4f}")

    # ── Step 4: Run inference ──
    for split_name, loader in loaders.items():
        run_inference(model, loader, device, split_name=split_name, max_images=args.max_images)

    logger.info(f"\n{'='*60}")
    logger.info("Inference complete!")
    logger.info(f"Generated captions saved in: {OUTPUT_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
