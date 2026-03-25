# autoruneterra

Educational data collection tool for clicker game automation research.

## Setup

```
pip install -r requirements.txt
```

## Usage

1. Configure `config.json` with your game window position/size and key shortcuts.
2. Run `python datacollector.py`
3. Press the capture key (default: `ctrl`) to screenshot the game and enter click-capture mode.
4. Click on the game — the screenshot is saved with normalized click coordinates.
5. Press `esc` to quit.

Saved files go to `click_dataset_folder/` as `{timestamp}_{x}_{y}.png` where x,y are 0-100 normalized positions.
