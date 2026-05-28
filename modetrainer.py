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
from trainer import IMAGENET_MEAN, IMAGENET_STD

MODE_TO_INDEX = {mode: i for i, mode in enumerate(MODES)}


#####  load data  ################################################################
class ModeDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # filename: {timestamp}_{x}_{y}.png where x,y are 0..100
        _, x_str, y_str = path.stem.split("_")
        mode = coord_to_mode(int(x_str), int(y_str))
        target = torch.tensor(MODE_TO_INDEX[mode], dtype=torch.long)

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_loaders(folder, batch_size=32, val_frac=0.2, seed=42, num_workers=0):
    files = sorted(Path(folder).glob("*.png"))
    rng = random.Random(seed)
    rng.shuffle(files)

    n_val = int(len(files) * val_frac)
    val_files = files[:n_val]
    train_files = files[n_val:]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = ModeDataset(train_files, transform=transform)
    val_ds = ModeDataset(val_files, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


#####  create model  ################################################################
def build_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Swap ResNet-50's 1000-way head for an N-way classifier over MODES.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


#####  train model  ################################################################
def train_model(model, train_loader, val_loader, device, save_path, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:>2}/{epochs}", leave=False)
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == targets).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="  validating", leave=False):
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == targets).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        # --- log + save best ---
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            marker = "  <- saved"
        print(
            f"epoch {epoch:>2}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{marker}"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    config = load_config()
    train_loader, val_loader = build_loaders(config["output_folder"])
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    model = build_model(num_classes=len(MODES)).to(device)
    save_path = f"mode_{config['model_name']}.pt"
    train_model(model, train_loader, val_loader, device, save_path=save_path, epochs=10)
