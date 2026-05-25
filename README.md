# autoruneterra

Educational data collection tool for clicker game automation research.

## Setup

PyTorch is required but not included in `requirements.txt` since installation varies by system. Install it first by following the [official guide](https://pytorch.org/get-started/locally/), then run:

```
pip install -r requirements.txt
```

## Game Settings

Before using the tool, set **Click UI** to **Click to Action** in the game's Options menu under General:

![Enable Click to Action](readme_assets/enable_click_to_action.png)

## Usage

1. Configure `config.json` with your game window position/size and key shortcuts.
2. Run `python datacollector.py`
3. Press the capture key (default: `ctrl`) to screenshot the game and enter click-capture mode.
4. Click on the game — the screenshot is saved with normalized click coordinates.
5. Press `esc` to quit.

Saved files go to `click_dataset_folder/` as `{timestamp}_{x}_{y}.png` where x,y are 0-100 normalized positions.

## Known Issues

- **Game must run on the primary monitor.** `PIL.ImageGrab.grab()` defaults to the primary monitor's coordinate space, so a window on a secondary monitor captures as a black image. Either move LoR to the primary display, or change the call in `capture_game_screenshot` to `ImageGrab.grab(bbox, all_screens=True)`.
