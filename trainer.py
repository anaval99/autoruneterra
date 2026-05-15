import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

#####  load data  ################################################################
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        target = torch.tensor(
            [int(x_str) / 100.0, int(y_str) / 100.0], dtype=torch.float32
        )

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
def build_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # ResNet-50's fc is Linear(2048, 1000) for ImageNet classes.
    # Swap it for Linear(2048, 2) so the head outputs (x, y).
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


#####  train model  ################################################################
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4, save_path="teemo.pt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:>2}/{epochs}", leave=False)
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)
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
            for images, targets in tqdm(val_loader, desc="  validating", leave=False):
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
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

    train_loader, val_loader = build_loaders("click_dataset_folder")
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    model = build_model().to(device)
    train_model(model, train_loader, val_loader, device, epochs=10)