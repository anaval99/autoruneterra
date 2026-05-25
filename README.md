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

## Collecting Data

1. Configure `config.json` with your game window position/size and key shortcuts.
2. Run `python datacollector.py`
3. Press the capture key (default: `shift`) to screenshot the game and enter click-capture mode.
4. Click on the game — the screenshot is saved with normalized click coordinates.
5. Press `esc` to quit.

Saved files go to `click_dataset_folder/` as `{timestamp}_{x}_{y}.png` where x,y are 0-100 normalized positions.

### Virtual and debug clicks

While in click-capture mode you can also use:

- `virtual_click_key` (default: `z`) — save the current screenshot tagged with the cursor's position without actually clicking the game.
- `debug_virtual_click_key` (default: `x`) — print the cursor's normalized coords without saving anything. Useful for measuring hitboxes.

## Training

The project trains two ResNet-50 fine-tunes that work together:

- **Coord regressor** (`{model_name}.pt`) — predicts the normalized click target `(x, y)` from a screenshot, conditioned on the action mode. Trained by `trainer.py`.
- **Mode classifier** (`mode_{model_name}.pt`) — predicts which action mode the screen is asking for. Trained by `modetrainer.py`.

```
python trainer.py        # -> {model_name}.pt
python modetrainer.py    # -> mode_{model_name}.pt
```

Both read `model_name` from `config.json` (e.g. `darius` → `darius.pt`, `mode_darius.pt`).

The regressor takes `(image, mode_onehot)` as input — the ResNet backbone produces a 2048-dim feature, the mode one-hot is concatenated onto it, and the final `Linear(2048 + N_modes, 2)` head outputs `(x, y)`. At inference the mode classifier runs first and its prediction is fed into the regressor, so the coord head focuses on the right region of the screen.

### Action modes

Mode labels are derived from the click coords in the filename via `coord_to_mode.py`. Each mode corresponds to a screen region:

- `confirmation` — diamond-shaped hitbox around the confirm button.
- `prepare_summon` — bottom band (`y > 91`).
- `prepare_battle` — mid band (`78 ≤ y ≤ 88`).
- `unknown` — anything else.

To add a new mode, **append** a new branch/region to `coord_to_mode.py` and add the label to the end of `MODES`. Never insert in the middle — class indices would shift and existing checkpoints would silently change meaning. Adding a mode changes the head size, so retrain both models afterwards.

### Cleaning the dataset

`unknown` samples are noise for the regressor (no consistent visual signature) and inflate accuracy for the classifier without teaching it anything useful. Move them out of training with:

```
python remove_unknown.py
```

This relocates them to `.unknown/` so they're out of `click_dataset_folder/` but not lost.

## Known Issues

- **Game must run on the primary monitor.** `PIL.ImageGrab.grab()` defaults to the primary monitor's coordinate space, so a window on a secondary monitor captures as a black image. Either move LoR to the primary display, or change the call in `capture_game_screenshot` to `ImageGrab.grab(bbox, all_screens=True)`.

  To switch primary displays on Windows, open **System → Display**, select the monitor you want as primary, and check **Make this my main display**:

  ![Set primary display](readme_assets/set_primary_display.png)
