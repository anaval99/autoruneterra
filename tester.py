import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from trainer import IMAGENET_MEAN, IMAGENET_STD, build_model


def load_teemo(weights_path, device):
    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, image_path, device):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)[0]
    return pred.cpu().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        default="1778714661_83_49",
        help="dataset filename without extension (e.g. 1778714661_83_49)",
    )
    parser.add_argument(
        "--folder",
        default="click_dataset_folder",
        help="folder containing the .png file",
    )
    parser.add_argument(
        "--recent",
        action="store_true",
        help="use the most recently modified .png in --folder (overrides --name)",
    )
    args = parser.parse_args()

    if args.recent:
        folder = Path(args.folder)
        files = list(folder.glob("*.png"))
        if not files:
            raise SystemExit(f"no .png files found in {folder}")
        image_path = max(files, key=lambda p: p.stat().st_mtime)
    else:
        image_path = Path(args.folder) / f"{args.name}.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_teemo("teemo.pt", device)

    pred_x, pred_y = predict(model, image_path, device)

    # ground truth from filename: {timestamp}_{x}_{y}.png  (x,y are 0..100)
    _, x_str, y_str = image_path.stem.split("_")
    true_x, true_y = int(x_str) / 100.0, int(y_str) / 100.0

    print(f"file: {image_path.name}")
    print(f"  predicted (norm): ({pred_x:.3f}, {pred_y:.3f})")
    print(f"  actual    (norm): ({true_x:.3f}, {true_y:.3f})")
    print(f"  error     (norm): ({abs(pred_x - true_x):.3f}, {abs(pred_y - true_y):.3f})")
    print()
    print("  in 0-100 scale:")
    print(f"    predicted: ({pred_x * 100:.1f}, {pred_y * 100:.1f})")
    print(f"    actual:    ({true_x * 100:.1f}, {true_y * 100:.1f})")
    print()
    print("  in 1920x1080 game pixels:")
    print(f"    predicted: ({pred_x * 1920:.0f}, {pred_y * 1080:.0f})")
    print(f"    actual:    ({true_x * 1920:.0f}, {true_y * 1080:.0f})")
