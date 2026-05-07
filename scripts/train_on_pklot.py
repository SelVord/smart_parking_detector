"""
Train MobileNetV2 on PKLot + your custom data.

Run from the project root:
    python scripts/train_on_pklot.py

No manual labeling needed — uses existing COCO annotations in data/pklot/.
"""

import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── Paths ─────────────────────────────────────────────────────
def _find_root() -> Path:
    """Walk up from this script until we find a directory that contains data/pklot."""
    candidate = Path(__file__).resolve().parent
    for _ in range(10):
        if (candidate / "data" / "pklot").exists():
            return candidate
        candidate = candidate.parent
    # Fallback: two levels up from script (worktree root)
    return Path(__file__).resolve().parent.parent

ROOT       = _find_root()
PKLOT_DIR  = ROOT / "data" / "pklot"
MYDATA_DIR = ROOT / "data" / "patches"
MODEL_PATH = ROOT / "models" / "cnn_classifier.pth"

print(ROOT, MODEL_PATH)

# ── Settings ──────────────────────────────────────────────────
PATCH_SIZE     = 64
MAX_PER_CLASS  = 30_000   # cap PKLot samples to keep training fast
EPOCHS         = 20
BATCH_SIZE     = 64
LR             = 0.001
VAL_RATIO      = 0.15
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── Dataset ───────────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Lazy patch dataset.  Each item is (img_path, bbox, label).
    The bbox is cropped and resized to PATCH_SIZE on __getitem__.
    """

    def __init__(self, samples: list, transform=None):
        # samples: list of (img_path, x, y, w, h, label)  label 0=empty 1=occupied
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, x, y, bw, bh, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        else:
            h, w = img.shape[:2]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + bw))
            y2 = min(h, int(y + bh))
            if x2 > x1 and y2 > y1:
                patch = cv2.resize(img[y1:y2, x1:x2], (PATCH_SIZE, PATCH_SIZE))
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            else:
                patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)

        pil = Image.fromarray(patch)
        if self.transform:
            pil = self.transform(pil)
        return pil, label


def _load_pklot_samples(split: str, max_per_class: int) -> list:
    """Load annotation records from one PKLot split, capped per class."""
    ann_path = PKLOT_DIR / split / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"  WARNING: {ann_path} not found, skipping {split}")
        return []

    with open(ann_path) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    img_dir    = PKLOT_DIR / split

    # category_id 1 = space-empty (0), 2 = space-occupied (1)
    cat_map = {1: 0, 2: 1}

    by_class = {0: [], 1: []}
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in cat_map:
            continue
        label    = cat_map[cat_id]
        fname    = id_to_file.get(ann["image_id"], "")
        img_path = img_dir / fname
        if not img_path.exists():
            continue
        bbox = [float(v) for v in ann["bbox"]]
        if bbox[2] < 5 or bbox[3] < 5:
            continue
        by_class[label].append((img_path, *bbox, label))

    # Cap and shuffle
    samples = []
    for label, items in by_class.items():
        random.shuffle(items)
        samples.extend(items[:max_per_class])
    return samples


def _load_mydata_samples() -> list:
    """Load pre-extracted patches from data/patches/ (your custom dataset)."""
    samples = []
    for label, cls in [("empty", 0), ("occupied", 1)]:
        folder = MYDATA_DIR / label
        if not folder.exists():
            continue
        for fname in folder.iterdir():
            if fname.suffix.lower() in (".jpg", ".jpeg", ".png"):
                # Store as (path, 0, 0, W, H, label) — full image is the patch
                img = cv2.imread(str(fname))
                if img is not None:
                    h, w = img.shape[:2]
                    samples.append((fname, 0, 0, w, h, cls))
    return samples


# ── Transforms ────────────────────────────────────────────────

_mean = [0.485, 0.456, 0.406]
_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

val_transform = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])


# ── Training ──────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Collect samples ─────────────────────────────────────
    print("\nLoading PKLot annotations…")
    pklot_train = _load_pklot_samples("train", MAX_PER_CLASS)
    pklot_val   = _load_pklot_samples("valid", MAX_PER_CLASS // 5)
    mydata      = _load_mydata_samples()

    print(f"  PKLot train samples : {len(pklot_train)}")
    print(f"  PKLot valid samples : {len(pklot_val)}")
    print(f"  Custom data samples : {len(mydata)}")

    if not pklot_train:
        print("\nERROR: No PKLot training data found.")
        print(f"Expected: {PKLOT_DIR / 'train' / '_annotations.coco.json'}")
        sys.exit(1)

    # Combine: all custom + PKLot; split PKLot into train/val
    all_train = pklot_train + mydata
    random.shuffle(all_train)

    # Use a portion of train as val if no dedicated val exists
    if pklot_val:
        val_samples   = pklot_val
        train_samples = all_train
    else:
        cut = int(len(all_train) * VAL_RATIO)
        val_samples, train_samples = all_train[:cut], all_train[cut:]

    empty_n    = sum(1 for s in train_samples if s[-1] == 0)
    occupied_n = sum(1 for s in train_samples if s[-1] == 1)
    print(f"\nTrain: {len(train_samples)} total  (empty={empty_n}, occupied={occupied_n})")
    print(f"Val  : {len(val_samples)}")

    # ── Datasets & loaders ──────────────────────────────────
    train_ds = PatchDataset(train_samples, train_transform)
    val_ds   = PatchDataset(val_samples,   val_transform)

    n_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=n_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=n_workers, pin_memory=(device.type == "cuda"))

    # ── Model ───────────────────────────────────────────────
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(1280, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Loop ────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += imgs.size(0)
        train_acc  = correct / total * 100
        train_loss = loss_sum / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)
        val_acc = val_correct / val_total * 100

        scheduler.step()
        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  => Saved best model (val_acc={val_acc:.1f}%)")

    print(f"\nDone. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    print(f"Project root : {ROOT}")
    print(f"PKLot data   : {PKLOT_DIR}")
    print(f"Custom data  : {MYDATA_DIR}")
    print(f"Model output : {MODEL_PATH}")
    print()
    train()
