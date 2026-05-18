import threading
import time

import keyboard
import pyautogui
import torch
from torchvision import transforms

from datacollector import capture_game_screenshot, find_game_window, load_config
from tester import load_teemo
from trainer import IMAGENET_MEAN, IMAGENET_STD

INPUT_SIZE = (448, 448)
WEIGHTS_PATH = "teemo.pt"


_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def infer(model, pil_image, device):
    tensor = _transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)[0]
    pred_x, pred_y = pred.cpu().tolist()
    pred_x = max(0.0, min(1.0, pred_x))
    pred_y = max(0.0, min(1.0, pred_y))
    return pred_x, pred_y


def run_one_cycle(model, device, gw):
    time.sleep(0.5)

    pyautogui.moveTo(gw["x"] + 1, gw["y"] + 1)
    time.sleep(0.5)

    screenshot = capture_game_screenshot(gw)
    screenshot = screenshot.resize(INPUT_SIZE)

    pred_x, pred_y = infer(model, screenshot, device)

    click_x = gw["x"] + round(pred_x * gw["width"])
    click_y = gw["y"] + round(pred_y * gw["height"])
    pyautogui.moveTo(click_x, click_y)

    time.sleep(0.5)
    pyautogui.click()

    print(f"[player] pred=({pred_x:.3f},{pred_y:.3f}) click=({click_x},{click_y})")


def main():
    config = load_config()
    window_title = config["game_window"]["title"]

    gw = find_game_window(window_title)
    if not gw:
        print(f"[error] Could not find window: '{window_title}'")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[player] device: {device}")
    print(f"[player] loading model from {WEIGHTS_PATH}...")
    model = load_teemo(WEIGHTS_PATH, device)
    print(f"[player] model loaded.")
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
            run_one_cycle(model, device, gw)
        else:
            time.sleep(0.05)

    print("Exiting.")


if __name__ == "__main__":
    main()
