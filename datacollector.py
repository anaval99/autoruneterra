import json
import math
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import keyboard
import mouse
import pyautogui
import pygetwindow
from PIL import ImageGrab


GAME_DATA_URL = "http://localhost:21337/positional-rectangles"


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def load_setlite(path="setlite.json"):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def fetch_card_summaries(setlite, timeout=2.0):
    """Fetch live game state and return a list of {cardCode, cost, attack, type}.

    Cards whose cardCode is not in setlite (e.g. "face") are skipped.
    """
    with urllib.request.urlopen(GAME_DATA_URL, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    summaries = []
    for rect in data.get("Rectangles", []):
        code = rect.get("CardCode")
        card = setlite.get(code)
        if card is None:
            continue
        summaries.append({
            "cardCode": code,
            "cost": card.get("cost"),
            "attack": card.get("attack"),
            "type": card.get("type"),
        })
    return summaries


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
    virtual_click_key = config["shortcuts"]["virtual_click_key"]
    debug_virtual_click_key = config["shortcuts"]["debug_virtual_click_key"]
    output_res = (config["output_resolution"]["width"], config["output_resolution"]["height"])
    output_folder = Path(config["output_folder"])
    output_folder.mkdir(exist_ok=True)
    setlite = load_setlite()

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

    def do_save(screenshot, filepath):
        screenshot.resize(output_res).save(filepath)
        print(f"[saved] {filepath.name}")

    def on_virtualclick(is_debug=False):
        if not clickcapture_mode or screenshot_in_memory is None:
            print("[virtual] No screenshot in memory — press capture key first.")
            return

        abs_x, abs_y = pyautogui.position()
        rel_x = abs_x - gw["x"]
        rel_y = abs_y - gw["y"]

        if rel_x < 0 or rel_y < 0 or rel_x > gw["width"] or rel_y > gw["height"]:
            print("[virtual] Mouse outside game window, ignored.")
            return

        norm_x = normalize_coord(rel_x, gw["width"])
        norm_y = normalize_coord(rel_y, gw["height"])
        timestamp = int(time.time())

        filename = f"{timestamp}_{norm_x}_{norm_y}.png"
        if not is_debug:
            stem = f"{timestamp}_{norm_x}_{norm_y}"
            try:
                summaries = fetch_card_summaries(setlite)
            except Exception as e:
                print(f"[abort] Failed to fetch game data, image not saved: {e}")
                return

            json_path = output_folder / f"{stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2)
            print(f"[saved] {json_path.name}  ({len(summaries)} cards)")

            filepath = output_folder / filename
            do_save(screenshot_in_memory, filepath)

        tag = "debug" if is_debug else "virtual"
        print(f"[{tag}] {filename}  (pixel: {rel_x},{rel_y} -> norm: {norm_x},{norm_y})")

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
    keyboard.on_press_key(virtual_click_key, lambda _: on_virtualclick())
    keyboard.on_press_key(debug_virtual_click_key, lambda _: on_virtualclick(is_debug=True))
    # mouse.hook(on_click) # We will use keyboard shortcuts for virtual clicks instead of real mouse clicks to avoid issues with hover states and timing.

    print("Listening for events...")
    keyboard.wait("esc")
    print("Exiting.")


if __name__ == "__main__":
    main()
