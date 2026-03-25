# ApexAI: F1 Winner Predictor

This project downloads Formula 1 race results with the
[FastF1](https://github.com/theOehrly/Fast-F1) library and trains a Gradient
Boosting model to estimate each driver's chance of winning the next race. The
model uses rolling performance features — recent win rate, podium rate, average
finish position, head-to-head vs teammate, driver experience, and team form — on
top of cumulative championship points. `RandomizedSearchCV` tunes
hyperparameters over 12 iterations with 5-fold CV. Includes a race-themed GUI
with track visualization, circuit-specific animations, team radio playback, and
supports the 2026 driver lineup.

## Setup

```bash
pip install -r requirements.txt
```

The first run will download timing data from the official F1 API and cache it in
`cache/`. If `data.csv` is not present, race results from 2022–2025 are
downloaded to build the training dataset.

## Usage

### GUI (recommended)

```bash
python app.py
```

Launch the race-themed interface. Click **RUN PREDICTIONS** to fetch data, train
the model, and see win probabilities for the next race with a podium-style
layout and full grid.

### Command line

```bash
python predict_winner.py
```

Prints the predicted winner for the last race, the upcoming round, and overall
model accuracy.

## Team logos (optional)

To use official team logos instead of colored badges:

1. Run `python fetch_logos.py` to download logos to `logos/`
2. Or add PNG files manually: `logos/redbull.png`, `logos/ferrari.png`, etc.

Without logos, the app shows colored badges with team initials.

## Building a Mac Executable

To create a standalone Mac `.app`:

1. Install PyInstaller: `pip install pyinstaller`
2. Run: `./build_mac.sh`

The app will be at `dist/F1 Winner Predictor.app`. You can drag it to
Applications. Data and cache are stored in
`~/Library/Application Support/F1 Winner Predictor/`.

## Data Source

Race and timing data is retrieved via `fastf1`, which accesses the official F1
live timing API.

## Screenshot

![ApexAI GUI](screenshot.png)
