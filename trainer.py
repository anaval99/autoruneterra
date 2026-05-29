import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from coord_to_mode import MODES, coord_to_mode
from datacollector import load_config
from get_manastone import get_manastones

#####  load data  ################################################################
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODE_TO_INDEX = {mode: i for i, mode in enumerate(MODES)}

# Per-card features:
#   [cost/COST_NORM, attack/ATTACK_NORM, is_unit,
#    topLeftX, topLeftY, width, height, localPlayer]
# cost/attack norms picked from setlite range (cost max 17, attack max 30) with
# some headroom. Spatial fields are pre-normalized in the sidecar JSON, so they
# just pass through here.
CARD_FEAT_DIM = 8
CARD_EMBED_DIM = 16
COST_NORM = 20.0
ATTACK_NORM = 30.0
# LoR base mana caps at 10; clamp+normalize so OCR misreads (e.g. 11+) don't
# blow up the feature. None -> 0 (OCR failure treated as "no info").
MANA_NORM = 10.0


def summaries_to_tensor(summaries):
    """Convert a list of card-summary dicts (see fetch_card_summaries) to a (N, 8) tensor."""
    rows = []
    for c in summaries:
        cost = c.get("cost")
        attack = c.get("attack")
        rows.append([
            (0.0 if cost is None else float(cost)) / COST_NORM,
            (0.0 if attack is None else float(attack)) / ATTACK_NORM,
            float(c.get("type") or 0),
            float(c.get("topLeftX") or 0),
            float(c.get("topLeftY") or 0),
            float(c.get("width") or 0),
            float(c.get("height") or 0),
            float(c.get("localPlayer") or 0),
        ])
    if not rows:
        return torch.zeros((0, CARD_FEAT_DIM), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def parse_card_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return summaries_to_tensor(json.load(f))


def pad_cards(card_tensors):
    """Pad a list of (Ni, 3) tensors to (B, max_N, 3) + build a (B, max_N) mask."""
    max_n = max((t.shape[0] for t in card_tensors), default=0)
    max_n = max(max_n, 1)  # avoid zero-length dim
    cards = torch.zeros(len(card_tensors), max_n, CARD_FEAT_DIM)
    mask = torch.zeros(len(card_tensors), max_n)
    for i, t in enumerate(card_tensors):
        n = t.shape[0]
        if n > 0:
            cards[i, :n] = t
            mask[i, :n] = 1.0
    return cards, mask


class ClickDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # filename: {timestamp}_{x}_{y}.png where x,y are 0..100
        _, x_str, y_str = path.stem.split("_")
        x_norm = int(x_str)
        y_norm = int(y_str)
        target = torch.tensor(
            [x_norm / 100.0, y_norm / 100.0], dtype=torch.float32
        )

        mode = coord_to_mode(x_norm, y_norm)
        mode_onehot = torch.zeros(len(MODES), dtype=torch.float32)
        mode_onehot[MODE_TO_INDEX[mode]] = 1.0

        raw = Image.open(path).convert("RGB")
        image = self.transform(raw) if self.transform is not None else raw

        cards = parse_card_json(path.with_suffix(".json"))
        return image, raw, cards, mode_onehot, target


def _click_collate(batch):
    images = torch.stack([b[0] for b in batch])
    raws = [b[1] for b in batch]
    cards, mask = pad_cards([b[2] for b in batch])
    modes = torch.stack([b[3] for b in batch])
    targets = torch.stack([b[4] for b in batch])
    mana_vals = get_manastones(raws)
    mana = torch.tensor(
        [min(max(m or 0, 0), int(MANA_NORM)) / MANA_NORM for m in mana_vals],
        dtype=torch.float32,
    ).unsqueeze(1)
    return images, cards, mask, modes, mana, targets


def split_dataset_files(folder, val_frac=0.2, seed=42):
    """Find all (png, json) pairs in `folder` and produce a deterministic train/val split.

    Single source of truth for both trainer.py and modetrainer.py — calling this from both
    keeps the coord regressor and mode classifier on matching train/val sides, so the mode
    classifier can't be evaluated on samples the coord regressor has seen (or vice versa).
    """
    files = [p for p in sorted(Path(folder).glob("*.png")) if p.with_suffix(".json").exists()]
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(len(files) * val_frac)
    return files[n_val:], files[:n_val]


def build_loaders(folder, input_size, batch_size=32, val_frac=0.2, seed=42, num_workers=0):
    """input_size is (width, height) — matches config.output_resolution."""
    train_files, val_files = split_dataset_files(folder, val_frac, seed)

    transform = transforms.Compose(
        [
            transforms.Resize((input_size[1], input_size[0])),  # torchvision expects (h, w)
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = ClickDataset(train_files, transform=transform)
    val_ds = ClickDataset(val_files, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=_click_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=_click_collate,
    )
    return train_loader, val_loader


#####  create model  ################################################################
class CardEncoder(nn.Module):
    """Permutation-invariant encoder: project per-card features, then attend with a learned query.

    A sum-pool collapses every card into a single aggregate (preserves totals/counts but loses
    per-card identity — "the 2-mana card is here" gets averaged away). A learned query attending
    over the cards lets the encoder pick out specific cards by their features.
    """

    def __init__(self, in_dim=CARD_FEAT_DIM, hidden=CARD_EMBED_DIM, n_heads=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, num_heads=n_heads, batch_first=True)
        self.out_dim = hidden

    def forward(self, cards, mask):
        # cards: (B, N, in_dim), mask: (B, N) with 1 = real card, 0 = padding
        h = self.proj(cards)
        q = self.query.expand(h.size(0), -1, -1)
        key_padding_mask = mask == 0
        # If every position is padded (sample with no cards), MHA would NaN — disable the mask
        # for those rows so attention runs over the zero-valued keys instead.
        all_padded = key_padding_mask.all(dim=1, keepdim=True)
        key_padding_mask = key_padding_mask & ~all_padded
        out, _ = self.attn(q, h, h, key_padding_mask=key_padding_mask)
        return out.squeeze(1)


class ClickModel(nn.Module):
    """ResNet-18 image features + card-set features + mana + mode one-hot -> (x, y)."""

    def __init__(self, num_modes):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.card_encoder = CardEncoder()
        self.head = nn.Linear(in_features + self.card_encoder.out_dim + 1 + num_modes, 2)

    def forward(self, image, cards, card_mask, mana, mode_onehot):
        img_feat = self.backbone(image)
        card_feat = self.card_encoder(cards, card_mask)
        return self.head(torch.cat([img_feat, card_feat, mana, mode_onehot], dim=1))


def build_model():
    return ClickModel(num_modes=len(MODES))


#####  train model  ################################################################
def train_model(model, train_loader, val_loader, device, save_path, epochs=10, lr=1/30000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:>2}/{epochs}", leave=False)
        for images, cards, card_mask, modes, mana, targets in pbar:
            images = images.to(device)
            cards = cards.to(device)
            card_mask = card_mask.to(device)
            modes = modes.to(device)
            mana = mana.to(device)
            targets = targets.to(device)

            preds = model(images, cards, card_mask, mana, modes)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader.dataset)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, cards, card_mask, modes, mana, targets in tqdm(val_loader, desc="  validating", leave=False):
                images = images.to(device)
                cards = cards.to(device)
                card_mask = card_mask.to(device)
                modes = modes.to(device)
                mana = mana.to(device)
                targets = targets.to(device)
                preds = model(images, cards, card_mask, mana, modes)
                loss = criterion(preds, targets)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        # --- log + save ---
        torch.save(model.state_dict(), save_path)
        print(f"epoch {epoch:>2}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  <- saved")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    config = load_config()
    input_size = (config["output_resolution"]["width"], config["output_resolution"]["height"])
    train_loader, val_loader = build_loaders(config["output_folder"], input_size, batch_size=16)
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    model = build_model().to(device)
    save_path = f"{config['model_name']}.pt"
    train_model(model, train_loader, val_loader, device, save_path=save_path, epochs=20)