import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from coord_to_mode import MODES, coord_to_mode
from datacollector import load_config
from trainer import (
    CardEncoder,
    IMAGENET_MEAN,
    IMAGENET_STD,
    pad_cards,
    parse_card_json,
    split_dataset_files,
)

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

        cards = parse_card_json(path.with_suffix(".json"))
        return image, cards, target


def _mode_collate(batch):
    images = torch.stack([b[0] for b in batch])
    cards, mask = pad_cards([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])
    return images, cards, mask, targets


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

    train_ds = ModeDataset(train_files, transform=transform)
    val_ds = ModeDataset(val_files, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=_mode_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=_mode_collate,
    )
    return train_loader, val_loader


#####  create model  ################################################################
class ModeModel(nn.Module):
    """ResNet-18 image features + card-set features -> mode logits."""

    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.card_encoder = CardEncoder()
        self.head = nn.Linear(in_features + self.card_encoder.out_dim, num_classes)

    def forward(self, image, cards, card_mask):
        img_feat = self.backbone(image)
        card_feat = self.card_encoder(cards, card_mask)
        return self.head(torch.cat([img_feat, card_feat], dim=1))


def build_model(num_classes):
    return ModeModel(num_classes=num_classes)


#####  train model  ################################################################
def train_model(model, train_loader, val_loader, device, save_path, epochs=10, lr=1/30000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:>2}/{epochs}", leave=False)
        for images, cards, card_mask, targets in pbar:
            images = images.to(device)
            cards = cards.to(device)
            card_mask = card_mask.to(device)
            targets = targets.to(device)

            logits = model(images, cards, card_mask)
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
            for images, cards, card_mask, targets in tqdm(val_loader, desc="  validating", leave=False):
                images = images.to(device)
                cards = cards.to(device)
                card_mask = card_mask.to(device)
                targets = targets.to(device)
                logits = model(images, cards, card_mask)
                loss = criterion(logits, targets)
                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == targets).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        # --- log + save ---
        torch.save(model.state_dict(), save_path)
        print(
            f"epoch {epoch:>2}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  <- saved"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    config = load_config()
    input_size = (config["output_resolution"]["width"], config["output_resolution"]["height"])
    train_loader, val_loader = build_loaders(config["output_folder"], input_size)
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    model = build_model(num_classes=len(MODES)).to(device)
    save_path = f"mode_{config['model_name']}.pt"
    train_model(model, train_loader, val_loader, device, save_path=save_path, epochs=10)
