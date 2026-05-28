import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from coord_to_mode import MODES, coord_to_mode
from datacollector import load_config

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

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        cards = parse_card_json(path.with_suffix(".json"))
        return image, cards, mode_onehot, target


def _click_collate(batch):
    images = torch.stack([b[0] for b in batch])
    cards, mask = pad_cards([b[1] for b in batch])
    modes = torch.stack([b[2] for b in batch])
    targets = torch.stack([b[3] for b in batch])
    return images, cards, mask, modes, targets


def build_loaders(folder, input_size, batch_size=32, val_frac=0.2, seed=42, num_workers=0):
    """input_size is (width, height) — matches config.output_resolution."""
    files = [p for p in sorted(Path(folder).glob("*.png")) if p.with_suffix(".json").exists()]
    rng = random.Random(seed)
    rng.shuffle(files)

    n_val = int(len(files) * val_frac)
    val_files = files[:n_val]
    train_files = files[n_val:]

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
    """Permutation-invariant encoder: per-card MLP, then masked sum across cards."""

    def __init__(self, in_dim=CARD_FEAT_DIM, hidden=CARD_EMBED_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.out_dim = hidden

    def forward(self, cards, mask):
        # cards: (B, N, in_dim), mask: (B, N)
        h = self.mlp(cards) * mask.unsqueeze(-1)
        return h.sum(dim=1)


class ClickModel(nn.Module):
    """ResNet-50 image features + card-set features + mode one-hot -> (x, y)."""

    def __init__(self, num_modes):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.card_encoder = CardEncoder()
        self.head = nn.Linear(in_features + self.card_encoder.out_dim + num_modes, 2)

    def forward(self, image, cards, card_mask, mode_onehot):
        img_feat = self.backbone(image)
        card_feat = self.card_encoder(cards, card_mask)
        return self.head(torch.cat([img_feat, card_feat, mode_onehot], dim=1))


def build_model():
    return ClickModel(num_modes=len(MODES))


#####  train model  ################################################################
def train_model(model, train_loader, val_loader, device, save_path, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:>2}/{epochs}", leave=False)
        for images, cards, card_mask, modes, targets in pbar:
            images = images.to(device)
            cards = cards.to(device)
            card_mask = card_mask.to(device)
            modes = modes.to(device)
            targets = targets.to(device)

            preds = model(images, cards, card_mask, modes)
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
            for images, cards, card_mask, modes, targets in tqdm(val_loader, desc="  validating", leave=False):
                images = images.to(device)
                cards = cards.to(device)
                card_mask = card_mask.to(device)
                modes = modes.to(device)
                targets = targets.to(device)
                preds = model(images, cards, card_mask, modes)
                loss = criterion(preds, targets)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        # --- log + save best ---
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            marker = "  <- saved"
        print(f"epoch {epoch:>2}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    config = load_config()
    input_size = (config["output_resolution"]["width"], config["output_resolution"]["height"])
    train_loader, val_loader = build_loaders(config["output_folder"], input_size)
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    model = build_model().to(device)
    save_path = f"{config['model_name']}.pt"
    train_model(model, train_loader, val_loader, device, save_path=save_path, epochs=10)