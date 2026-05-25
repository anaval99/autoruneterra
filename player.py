import threading
import time

import keyboard
import pyautogui
import torch
import torch.nn.functional as F
from torchvision import transforms

from coord_to_mode import MODES
from datacollector import capture_game_screenshot, find_game_window, load_config
from modetrainer import build_model as build_mode_model
from trainer import IMAGENET_MEAN, IMAGENET_STD, build_model as build_coord_model

INPUT_SIZE = (448, 448)


_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_weights(model, weights_path, device):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def infer(coord_model, mode_model, pil_image, device):
    tensor = _transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        mode_idx = int(mode_model(tensor).argmax(1).item())
        mode_onehot = F.one_hot(
            torch.tensor([mode_idx], device=device), num_classes=len(MODES)
        ).float()
        pred = coord_model(tensor, mode_onehot)[0]
    pred_x, pred_y = pred.cpu().tolist()
    pred_x = max(0.0, min(1.0, pred_x))
    pred_y = max(0.0, min(1.0, pred_y))
    return pred_x, pred_y, MODES[mode_idx]


cycle_interval = 0.5


def normcoord_to_pixel(norm_x, norm_y, gw):
    pixel_x = round(norm_x * gw["width"])
    pixel_y = round(norm_y * gw["height"])
    return pixel_x, pixel_y


def run_one_cycle(coord_model, mode_model, device, gw):
    time.sleep(cycle_interval)  # to avoid too fast clicking, adjust as needed

    pyautogui.moveTo(gw["x"] + 1, gw["y"] + 1)
    time.sleep(cycle_interval)  # small delay to ensure mouse move is registered and any hover effects are cleared

    screenshot = capture_game_screenshot(gw)
    screenshot = screenshot.resize(INPUT_SIZE)
    time.sleep(cycle_interval)  # small delay to ensure screenshot is captured properly

    pred_x, pred_y, mode = infer(coord_model, mode_model, screenshot, device)
    click_x, click_y = normcoord_to_pixel(pred_x, pred_y, gw)
    pyautogui.moveTo(click_x, click_y)

    time.sleep(cycle_interval)  # small delay before clicking to ensure mouse move is registered
    pyautogui.click()

    print(
        f"[player] mode={mode}  pred=({pred_x:.3f},{pred_y:.3f})  click=({click_x},{click_y})"
    )


def main():
    config = load_config()
    window_title = config["game_window"]["title"]
    model_name = config["model_name"]
    coord_weights = f"{model_name}.pt"
    mode_weights = f"mode_{model_name}.pt"

    gw = find_game_window(window_title)
    if not gw:
        print(f"[error] Could not find window: '{window_title}'")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[player] device: {device}")
    print(f"[player] loading coord model from {coord_weights}...")
    coord_model = load_weights(build_coord_model(), coord_weights, device)
    print(f"[player] loading mode model from {mode_weights}...")
    mode_model = load_weights(
        build_mode_model(num_classes=len(MODES)), mode_weights, device
    )
    print(f"[player] models loaded.")
    print(f"[player] game window: {gw['width']}x{gw['height']} at ({gw['x']},{gw['y']})")

    running = threading.Event()
    stop = threading.Event()

    def toggle_running(_):
        if running.is_set():
            running.clear()
            print("[player] paused")
        else:
            running.set()
            print("[player] running")

    def request_stop(_):
        stop.set()

    keyboard.on_press_key("shift", toggle_running)
    keyboard.on_press_key("esc", request_stop)

    print("Press SHIFT to start/stop the loop. Press ESC to quit.")

    while not stop.is_set():
        if running.is_set():
            run_one_cycle(coord_model, mode_model, device, gw)
        else:
            time.sleep(0.05)

    print("Exiting.")


if __name__ == "__main__":
    main()
