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
        return image, mode_onehot, target


def build_loaders(folder, input_size, batch_size=32, val_frac=0.2, seed=42, num_workers=0):
    """input_size is (width, height) — matches config.output_resolution."""
    files = sorted(Path(folder).glob("*.png"))
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
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


#####  create model  ################################################################
class ClickModel(nn.Module):
    """ResNet-50 backbone + mode one-hot concatenated into the regression head."""

    def __init__(self, num_modes):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_features + num_modes, 2)

    def forward(self, image, mode_onehot):
        feat = self.backbone(image)
        return self.head(torch.cat([feat, mode_onehot], dim=1))


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
        for images, modes, targets in pbar:
            images = images.to(device)
            modes = modes.to(device)
            targets = targets.to(device)

            preds = model(images, modes)
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
            for images, modes, targets in tqdm(val_loader, desc="  validating", leave=False):
                images = images.to(device)
                modes = modes.to(device)
                targets = targets.to(device)
                preds = model(images, modes)
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