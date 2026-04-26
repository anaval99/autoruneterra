import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import keyboard
import mouse
import pyautogui
import pygetwindow
from PIL import ImageGrab


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def find_game_window(title):
    """Find the game window by title and return its position/size dict."""
    windows = pygetwindow.getWindowsWithTitle(title)
    if not windows:
        return None
    win = windows[0]
    return {"x": win.left, "y": win.top, "width": win.width, "height": win.height}


def capture_game_screenshot(gw):
    """Capture the game window region and return the PIL image."""
    bbox = (gw["x"], gw["y"], gw["x"] + gw["width"], gw["y"] + gw["height"])
    return ImageGrab.grab(bbox)


def normalize_coord(value, max_value):
    """Normalize a pixel coordinate to 0-100 range with rounding rules.

    < 10 → ceil, > 90 → floor, otherwise round.
    """
    pct = (value / max_value) * 100
    if pct < 10:
        return math.ceil(pct)
    elif pct > 90:
        return math.floor(pct)
    else:
        return round(pct)



def main():
    config = load_config()
    window_title = config["game_window"]["title"]
    capture_key = config["shortcuts"]["capture_key"]
    idle_key = config["shortcuts"]["idle_key"]
    output_folder = Path(config["output_folder"])
    output_folder.mkdir(exist_ok=True)

    gw = find_game_window(window_title)
    if not gw:
        print(f"[error] Could not find window: '{window_title}'")
        return

    screenshot_in_memory = None
    clickcapture_mode = False

    print(f"Data Collector started.")
    print(f"  Capture key : {capture_key}")
    print(f"  Game window : {gw['width']}x{gw['height']} at ({gw['x']},{gw['y']})")
    print(f"  Output      : {output_folder}/")
    print(f"Press '{capture_key}' to capture a screenshot and enter click-capture mode.")
    print("Press 'esc' to quit.\n")

    def on_capture():
        nonlocal screenshot_in_memory, clickcapture_mode
        # Move mouse to top-left corner (1,1) to clear any hover effects
        pyautogui.moveTo(gw["x"] + 1, gw["y"] + 1)
        time.sleep(0.5)
        screenshot_in_memory = capture_game_screenshot(gw)
        clickcapture_mode = True
        print("[capture] Screenshot stored. Click-capture mode ON.")
        
    def on_idle():
        current_screenshot = capture_game_screenshot(gw)
        timestamp = int(time.time())
        filename = f"{timestamp}_{1}_{1}.png"
        filepath = output_folder / filename
        print(f"[idle] Captured idle screenshot: {filename}")
        do_save(current_screenshot, filepath)
        
    def do_save(screenshot, filepath):
        screenshot.save(filepath)
        print(f"[saved] {filepath.name}")

    def on_click(event):
        nonlocal screenshot_in_memory, clickcapture_mode
        if not clickcapture_mode or screenshot_in_memory is None:
            return
        if not isinstance(event, mouse.ButtonEvent) or event.event_type != "up":
            return

        # Get click position relative to game window
        # sample values for abs_x, abs_y = (600, 400)
        # sample values for gw x, y = (100, 230)
        # rel_x = 600 - 100 = 500, where gw[x] is the left edge of the game window
        # rel_y = 400 - 230 = 170, where gw[y] is the top edge of the game window
        abs_x, abs_y = pyautogui.position()
        rel_x = abs_x - gw["x"]
        rel_y = abs_y - gw["y"]

        # Ignore clicks outside the game window
        if rel_x < 0 or rel_y < 0 or rel_x > gw["width"] or rel_y > gw["height"]:
            print("[click] Outside game window, ignored.")
            return

        norm_x = normalize_coord(rel_x, gw["width"])
        norm_y = normalize_coord(rel_y, gw["height"])
        timestamp = int(time.time())

        filename = f"{timestamp}_{norm_x}_{norm_y}.png"
        filepath = output_folder / filename
        do_save(screenshot_in_memory, filepath)

        print(f"[saved] {filename}  (pixel: {rel_x},{rel_y} -> norm: {norm_x},{norm_y})")
        clickcapture_mode = False
        screenshot_in_memory = None
        print("[mode] CLICKSAVED — press capture key to take a new screenshot.")

    keyboard.on_press_key(capture_key, lambda _: on_capture())
    keyboard.on_press_key(idle_key, lambda _: on_idle())
    mouse.hook(on_click)

    print("Listening for events...")
    keyboard.wait("esc")
    print("Exiting.")


if __name__ == "__main__":
    main()
