"""
Beginner-friendly Google Colab smoke test for BLIP-2 on Flickr8k.

Copy this entire file into a single Colab cell and run it.

What it does:
1. Installs the minimum packages needed for this repo's BLIP-2 code path.
2. Clones the repo branch used in this workspace.
3. Downloads Flickr8k from Kaggle after you upload kaggle.json.
4. Builds a tiny train/val/test split for a smoke test.
5. Loads BLIP-2 with PVT v2 b2 and applies Q-Former LoRA.
6. Trains for a short run so you can verify the model is alive.
7. Generates captions for 3 validation images.

This is intentionally a smoke test, not final training.
It uses:
- PVT v2 b2 vision encoder
- Q-Former LoRA
- flan-t5-small to keep Colab memory use low

If this runs end-to-end, your larger training setup is much less likely to fail
for basic plumbing reasons.
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import subprocess
import sys
import types
import warnings
from collections import defaultdict
from pathlib import Path


# -----------------------------
# User-editable settings
# -----------------------------
REPO_URL = "https://github.com/archi-dev1/lavis-ai.git"
REPO_BRANCH = "pre-training-stable"
KAGGLE_DATASET = "adityajn105/flickr8k"

TRAIN_IMAGE_LIMIT = 512
VAL_IMAGE_LIMIT = 64
TEST_IMAGE_LIMIT = 64
CAPTIONS_PER_IMAGE = 1

IMAGE_SIZE = 224
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
NUM_WORKERS = 2
MAX_STEPS = 80
EPOCHS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
SAVE_DRIVE_COPY = False
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/blip2_flickr8k_smoke")

NUM_BEAMS = 3
MAX_LENGTH = 20
NO_REPEAT_NGRAM_SIZE = 2

WORKDIR = Path("/content")
REPO_DIR = WORKDIR / "lavis-ai"
DATA_ROOT = WORKDIR / "data" / "flickr8k"
OUTPUT_DIR = WORKDIR / "outputs" / "flickr8k_smoke"


def run(cmd, cwd=None):
    if isinstance(cmd, str):
        shell = True
        printable = cmd
    else:
        shell = False
        printable = " ".join(cmd)

    print(f"\n$ {printable}")
    subprocess.run(cmd, cwd=cwd, shell=shell, check=True)


def pip_install(*packages):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *packages]
    run(cmd)


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def install_dependencies():
    print("\n[1/8] Installing dependencies...")
    pip_install(
        "transformers==4.46.2",
        "peft==0.13.2",
        "timm==1.0.15",
        "omegaconf==2.3.0",
        "iopath",
        "fairscale",
        "sentencepiece",
        "einops",
        "kaggle",
        "pandas",
        "pillow",
    )


def clone_repo():
    print("\n[2/8] Cloning repo...")
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO_DIR)])
    else:
        print(f"Reusing existing repo at {REPO_DIR}")
    os.chdir(REPO_DIR)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))


def patch_repo_if_needed():
    print("\n[3/8] Patching repo for Colab smoke test if needed...")
    target = REPO_DIR / "lavis" / "models" / "blip2_models" / "blip2_t5.py"
    text = target.read_text()

    if "no_repeat_ngram_size=0" not in text:
        old_signature = """    def generate(\n        self,\n        samples,\n        use_nucleus_sampling=False,\n        num_beams=5,\n        max_length=30,\n        min_length=1,\n        top_p=0.9,\n        repetition_penalty=1.0,\n        length_penalty=1.0,\n        num_captions=1,\n        temperature=1,\n    ):"""
        new_signature = """    def generate(\n        self,\n        samples,\n        use_nucleus_sampling=False,\n        num_beams=5,\n        max_length=30,\n        min_length=1,\n        top_p=0.9,\n        repetition_penalty=1.0,\n        length_penalty=1.0,\n        num_captions=1,\n        temperature=1,\n        no_repeat_ngram_size=0,\n    ):"""
        text = text.replace(old_signature, new_signature)

        old_call = """                repetition_penalty=repetition_penalty,\n                length_penalty=length_penalty,\n                num_return_sequences=num_captions,\n            )"""
        new_call = """                repetition_penalty=repetition_penalty,\n                length_penalty=length_penalty,\n                num_return_sequences=num_captions,\n                no_repeat_ngram_size=no_repeat_ngram_size,\n            )"""
        text = text.replace(old_call, new_call)
        target.write_text(text)
        print("Patched Blip2T5.generate() to accept no_repeat_ngram_size.")
    else:
        print("No patch needed.")


def setup_kaggle_credentials():
    print("\n[4/8] Setting up Kaggle credentials...")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        kaggle_json.chmod(0o600)
        print(f"Found existing {kaggle_json}")
        return

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        try:
            from google.colab import userdata

            username = username or userdata.get("KAGGLE_USERNAME")
            key = key or userdata.get("KAGGLE_KEY")
        except Exception:
            pass

    if username and key:
        kaggle_json.write_text(
            '{\n'
            f'  "username": "{username}",\n'
            f'  "key": "{key}"\n'
            '}\n'
        )
        kaggle_json.chmod(0o600)
        print(f"Created {kaggle_json} from environment or Colab Secrets")
        return

    try:
        from getpass import getpass

        print("No Kaggle credentials found yet.")
        print("Option 1: paste your Kaggle username and key now.")
        print("Option 2: press Enter on username and upload kaggle.json instead.")
        entered_username = input("Kaggle username: ").strip()
        entered_key = getpass("Kaggle key: ").strip() if entered_username else ""

        if entered_username and entered_key:
            kaggle_json.write_text(
                '{\n'
                f'  "username": "{entered_username}",\n'
                f'  "key": "{entered_key}"\n'
                '}\n'
            )
            kaggle_json.chmod(0o600)
            print(f"Saved credentials to {kaggle_json}")
            return
    except Exception:
        pass

    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError(
            "No Kaggle credentials found. Set KAGGLE_USERNAME/KAGGLE_KEY, use Colab Secrets, "
            "enter them interactively, or place kaggle.json in ~/.kaggle/."
        ) from exc

    print("Upload kaggle.json if you do not want to paste the username and key.")
    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        raise RuntimeError("kaggle.json was not uploaded.")

    kaggle_json.write_bytes(uploaded["kaggle.json"])
    kaggle_json.chmod(0o600)
    print(f"Saved credentials to {kaggle_json}")


def find_first(root, names):
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in names:
            return path
    return None


def find_images_dir(root):
    candidates = []
    for path in root.rglob("*"):
        if path.is_dir():
            count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.jpeg"))) + len(list(path.glob("*.png")))
            if count >= 100:
                candidates.append((count, path))
    if not candidates:
        raise RuntimeError("Could not find the Flickr8k images directory after download.")
    candidates.sort(reverse=True)
    return candidates[0][1]


def download_flickr8k():
    print("\n[5/8] Downloading Flickr8k from Kaggle...")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if not any(DATA_ROOT.rglob("*.jpg")) and not any(DATA_ROOT.rglob("*.png")):
        run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(DATA_ROOT), "--unzip"])
    else:
        print(f"Reusing existing dataset at {DATA_ROOT}")

    captions_file = find_first(
        DATA_ROOT,
        {
            "captions.txt",
            "captions.csv",
            "flickr8k.token.txt",
            "flickr8k_text.zip",
        },
    )
    images_dir = find_images_dir(DATA_ROOT)

    if captions_file is None:
        token_like = list(DATA_ROOT.rglob("*.txt")) + list(DATA_ROOT.rglob("*.csv"))
        if not token_like:
            raise RuntimeError("Could not find the Flickr8k captions file after download.")
        captions_file = token_like[0]

    print(f"Images directory: {images_dir}")
    print(f"Captions file:    {captions_file}")
    return images_dir, captions_file


def read_split_file(base_dir, filename):
    path = find_first(base_dir, {filename.lower()})
    if path is None:
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_flickr8k_captions(captions_file, images_dir):
    captions_by_image = defaultdict(list)
    suffix = captions_file.suffix.lower()

    if captions_file.name.lower() == "flickr8k.token.txt":
        for line in captions_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_part, caption = line.split("\t", 1)
            image_name = image_part.split("#", 1)[0].strip()
            captions_by_image[image_name].append(caption.strip())
    elif suffix == ".csv":
        import pandas as pd

        frame = pd.read_csv(captions_file)
        columns = {col.lower(): col for col in frame.columns}
        image_col = columns.get("image") or columns.get("filename") or frame.columns[0]
        caption_col = columns.get("caption") or columns.get("raw") or frame.columns[1]
        for _, row in frame.iterrows():
            captions_by_image[str(row[image_col]).strip()].append(str(row[caption_col]).strip())
    else:
        with captions_file.open("r", encoding="utf-8") as handle:
            sample = handle.read(2048)
            handle.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.reader(handle, dialect)
            rows = list(reader)
        if rows and rows[0] and rows[0][0].lower() in {"image", "filename"}:
            rows = rows[1:]
        for row in rows:
            if len(row) < 2:
                continue
            captions_by_image[row[0].strip()].append(row[1].strip())

    available = {}
    for image_name, captions in captions_by_image.items():
        image_path = images_dir / image_name
        if image_path.exists() and captions:
            available[image_name] = captions

    if not available:
        raise RuntimeError("No valid image/caption pairs were found in Flickr8k.")

    return available


def build_splits(base_dir, available_captions):
    train_names = read_split_file(base_dir, "Flickr_8k.trainImages.txt")
    val_names = read_split_file(base_dir, "Flickr_8k.devImages.txt")
    test_names = read_split_file(base_dir, "Flickr_8k.testImages.txt")

    if train_names and val_names and test_names:
        train_names = [name for name in train_names if name in available_captions][:TRAIN_IMAGE_LIMIT]
        val_names = [name for name in val_names if name in available_captions][:VAL_IMAGE_LIMIT]
        test_names = [name for name in test_names if name in available_captions][:TEST_IMAGE_LIMIT]
    else:
        names = sorted(available_captions.keys())
        random.shuffle(names)
        train_names = names[:TRAIN_IMAGE_LIMIT]
        val_names = names[TRAIN_IMAGE_LIMIT:TRAIN_IMAGE_LIMIT + VAL_IMAGE_LIMIT]
        test_names = names[
            TRAIN_IMAGE_LIMIT + VAL_IMAGE_LIMIT:
            TRAIN_IMAGE_LIMIT + VAL_IMAGE_LIMIT + TEST_IMAGE_LIMIT
        ]

    if len(train_names) < 32:
        raise RuntimeError("Too few training images found for the smoke test.")

    return train_names, val_names, test_names


def prepare_lavis_imports():
    for pkg in [
        "lavis",
        "lavis.datasets",
        "lavis.datasets.builders",
        "lavis.processors",
        "lavis.tasks",
        "lavis.runners",
        "lavis.models",
    ]:
        if pkg not in sys.modules:
            module = types.ModuleType(pkg)
            module.__path__ = [str(REPO_DIR / pkg.replace(".", "/"))]
            module.__package__ = pkg
            sys.modules[pkg] = module

    import lavis.common.registry  # noqa: F401
    import lavis.common.utils  # noqa: F401
    from lavis.models.base_model import BaseModel

    sys.modules["lavis.models"].BaseModel = BaseModel


def build_model(device):
    print("\n[6/8] Building BLIP-2 model with PVT + LoRA...")
    prepare_lavis_imports()

    from lavis.models.blip2_models.blip2 import (
        apply_lora_to_qformer,
        freeze_qformer_base,
        get_trainable_params_info,
    )
    from lavis.models.blip2_models.blip2_t5 import Blip2T5

    model = Blip2T5(
        vit_model="pvt_v2_b2",
        img_size=IMAGE_SIZE,
        num_query_token=16,
        t5_model="google/flan-t5-small",
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        prompt="a photo of ",
        max_txt_len=32,
    )

    model.Qformer = apply_lora_to_qformer(
        model.Qformer,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
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
    model.train()

    info = get_trainable_params_info(model)
    print(f"Trainable params: {info['trainable_params']:,}")
    print(f"Total params:     {info['total_params']:,}")
    print(f"Trainable pct:    {info['trainable_percentage']:.4f}%")
    return model


def build_transforms():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class Flickr8kCaptionDataset:
    def __init__(self, image_dir, captions_by_image, image_names, transform, prompt):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.prompt = prompt
        self.samples = []
        for image_name in image_names:
            for caption in captions_by_image[image_name][:CAPTIONS_PER_IMAGE]:
                self.samples.append((image_name, caption.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        image_name, caption = self.samples[index]
        image_path = self.image_dir / image_name
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        return {
            "image": image_tensor,
            "text_input": self.prompt,
            "text_output": caption,
            "image_name": image_name,
        }


def collate_fn(batch):
    import torch

    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "text_input": [item["text_input"] for item in batch],
        "text_output": [item["text_output"] for item in batch],
        "image_name": [item["image_name"] for item in batch],
    }


def train_smoke_test(model, train_loader, device):
    print("\n[7/8] Running short smoke training...")
    import torch

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    global_step = 0
    loss_history = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        for batch_index, batch in enumerate(train_loader, start=1):
            global_step += 1
            batch["image"] = batch["image"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                outputs = model(batch)
                loss = outputs["loss"] / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if batch_index % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_value = float((loss.detach().cpu().item()) * GRAD_ACCUM_STEPS)
            loss_history.append(loss_value)

            if global_step == 1 or global_step % 10 == 0:
                avg_loss = sum(loss_history[-10:]) / min(len(loss_history), 10)
                print(f"step {global_step:03d} | loss={loss_value:.4f} | avg10={avg_loss:.4f}")

            if global_step >= MAX_STEPS:
                break

        if batch_index % GRAD_ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if global_step >= MAX_STEPS:
            break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "blip2_flickr8k_smoke.pt"
    torch.save({"model": model.state_dict(), "step": global_step}, checkpoint_path)
    print(f"\nSaved smoke-test checkpoint to {checkpoint_path}")

    if SAVE_DRIVE_COPY:
        DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        drive_path = DRIVE_OUTPUT_DIR / checkpoint_path.name
        shutil.copy2(checkpoint_path, drive_path)
        print(f"Copied checkpoint to {drive_path}")


def run_inference(model, image_dir, sample_image_names, device):
    print("\n[8/8] Generating sample captions...")
    import torch
    from PIL import Image

    transform = build_transforms()
    model.eval()

    with torch.no_grad():
        for image_name in sample_image_names[:3]:
            image_path = Path(image_dir) / image_name
            processed_image = None
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                processed_image = transform(image).unsqueeze(0).to(device)

                if tuple(processed_image.shape) != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
                    print(f"Shape mismatch for {image_name}")
                    print(f"Input shape:  {tuple(processed_image.shape)}")

                captions = model.generate(
                    {"image": processed_image, "prompt": "a photo of "},
                    use_nucleus_sampling=False,
                    num_beams=NUM_BEAMS,
                    max_length=MAX_LENGTH,
                    min_length=3,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                )

                caption = captions[0].strip() if isinstance(captions, list) else str(captions).strip()
                if not caption:
                    print(f"No caption returned for {image_name}; check tokenizer decoding.")
                    caption = "[empty-caption]"

                print(f"Image: {image_name}")
                print(f"Caption: {caption}")
                print()
            except Exception as exc:
                print(f"Image: {image_name}")
                print(f"Caption: ERROR - {exc}")
                print("generate() diagnostic checks:")
                print(f"  tokenizer present: {hasattr(model, 't5_tokenizer')}")
                print(f"  model device: {model.device}")
                print(
                    "  input shape: "
                    f"{tuple(processed_image.shape) if processed_image is not None else 'N/A'}"
                )
                print()

    model.train()


def main():
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    install_dependencies()
    clone_repo()
    patch_repo_if_needed()
    setup_kaggle_credentials()
    images_dir, captions_file = download_flickr8k()

    import torch
    from torch.utils.data import DataLoader

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type != "cuda":
        print("Warning: GPU runtime is strongly recommended in Colab.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} | VRAM: {total_mem_gb:.1f} GB")
        print(
            f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS} "
            f"({BATCH_SIZE} x grad accumulation {GRAD_ACCUM_STEPS})"
        )

    captions_by_image = parse_flickr8k_captions(captions_file, images_dir)
    train_names, val_names, test_names = build_splits(DATA_ROOT, captions_by_image)

    print(f"Train images: {len(train_names)}")
    print(f"Val images:   {len(val_names)}")
    print(f"Test images:  {len(test_names)}")

    transform = build_transforms()
    train_dataset = Flickr8kCaptionDataset(
        image_dir=images_dir,
        captions_by_image=captions_by_image,
        image_names=train_names,
        transform=transform,
        prompt="a photo of ",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = build_model(device)
    train_smoke_test(model, train_loader, device)
    run_inference(model, images_dir, val_names or test_names, device)

    print("Smoke test finished.")
    print(f"Checkpoint: {OUTPUT_DIR / 'blip2_flickr8k_smoke.pt'}")
    print("If captions print and loss updates, the pipeline is working.")


if __name__ == "__main__":
    main()