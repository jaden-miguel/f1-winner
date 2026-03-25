#!/usr/bin/env python3
"""
ApexAI – F1 winner prediction
"""
import io
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("tkinter is required.")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prediction import run_predictions, run_predictions_all_races, predict_with_standings, F1_POINTS
from team_colors import TEAM_COLORS
from team_logos import load_logo
from track_layouts import get_track

try:
    import f1radio
    HAS_RADIO = True
except ImportError:
    HAS_RADIO = False

# -- Theme: Black + Red / Carbon Fiber --
BG = "#080808"
BG_SURFACE = "#0e0e0f"
BG_CARD = "#121213"
BG_HOVER = "#1c1c1e"
BORDER = "#2a1216"
GOLD = "#d4232a"
GOLD_DIM = "#3a0e12"
GOLD_GLOW = "#ff3038"
WHITE = "#f0f0f0"
GRAY = "#8a8a90"
MUTED = "#58585e"
RED = "#d4232a"
GREEN = "#2ed87a"

# Carbon-fiber accent: thin red stripe on borders, dark weave texture on panels
CF_LIGHT = "#161617"
CF_DARK = "#0f0f10"


def tc(team: str) -> str:
    return TEAM_COLORS.get(team, "#555566")


def _make_carbon_fiber_img(w, h):
    """Generate a carbon fiber weave texture using PIL."""
    if not HAS_PIL:
        return None
    img = Image.new("RGB", (w, h), (12, 12, 13))
    pix = img.load()
    for y in range(h):
        for x in range(w):
            cell_x, cell_y = x % 6, y % 6
            if (cell_x < 3) == (cell_y < 3):
                pix[x, y] = (15, 15, 16)
            else:
                pix[x, y] = (11, 11, 12)
            if cell_x == 0 or cell_y == 0:
                pix[x, y] = (9, 9, 10)
    return img


ALGO_TEXT = """\
# ApexAI · Gradient Boosting Pipeline

features = [
  "Abbreviation",       # VER, HAM, NOR...
  "TeamName",           # Ferrari, McLaren...
  "GridPosition",       # Starting grid slot
  "RecentAvgPos",       # Avg finish, last 5
  "RecentWinRate",      # Win % in last 10
  "RecentPodiumRate",   # Podium % in last 10
  "DriverExperience",   # Career race count
  "HeadToHead",         # % beating teammate
  "TeamRecentForm",     # Team avg finish
  "DriverPointsBefore", # Cumulative driver pts
  "TeamPointsBefore",   # Cumulative team pts
]

preprocess = ColumnTransformer([
  OneHotEncoder → driver, team
  StandardScaler → numeric features
])

model = GradientBoostingClassifier(
  n_estimators = 200–500,
  max_depth    = 4–8,
  learning_rate= 0.05–0.15,
)

tuning = RandomizedSearchCV(
  n_iter=12, cv=5,
  scoring="f1",
)

output = model.predict_proba(X)[:, 1]
# → win probability per driver\
"""


SCENES = {
    # (vegetation_type, veg_color, veg_spacing, ground_color, [(features...)])
    "Albert Park":
        ("deciduous", "#1a3a1a", 18, "#0a120a",
         [("lake",), ("grandstand", 0.0, 1), ("grandstand", 0.45, -1)]),
    "Sakhir":
        (None, None, 0, "#120f08",
         [("dunes", 4), ("grandstand", 0.0, 1)]),
    "Jeddah Corniche":
        (None, None, 0, "#0a0a12",
         [("water", "left"), ("buildings", "right", 6), ("grandstand", 0.0, 1)]),
    "Suzuka":
        ("cherry", "#3a1a28", 15, "#0a120a",
         [("ferris", 0.92, 0.10), ("grandstand", 0.0, 1), ("grandstand", 0.6, -1)]),
    "Shanghai":
        ("deciduous", "#1a2a1a", 22, "#0a120a",
         [("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Miami Autodrome":
        ("palm", "#1a5a1a", 22, "#0a120a",
         [("water", "right"), ("skyline", "left", 5), ("grandstand", 0.0, 1)]),
    "Imola":
        ("deciduous", "#1a3a1a", 16, "#0a120a",
         [("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Monaco":
        (None, None, 0, "#0a0a12",
         [("water", "bottom"), ("yachts", 4), ("buildings", "top", 8), ("grandstand", 0.0, 1)]),
    "Barcelona-Catalunya":
        ("deciduous", "#2a3a1a", 22, "#0a100a",
         [("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Circuit Gilles Villeneuve":
        ("deciduous", "#1a3a1a", 16, "#0a120a",
         [("water", "right"), ("water", "left"), ("grandstand", 0.0, 1)]),
    "Red Bull Ring":
        ("pine", "#0a2a0a", 14, "#08100a",
         [("mountains", 5), ("grandstand", 0.0, 1), ("grandstand", 0.4, -1)]),
    "Silverstone":
        ("deciduous", "#1a301a", 20, "#0a100a",
         [("grandstand", 0.0, 1), ("grandstand", 0.35, -1), ("grandstand", 0.7, 1)]),
    "Spa-Francorchamps":
        ("pine", "#0a2a0a", 12, "#08100a",
         [("mountains", 4), ("grandstand", 0.0, 1)]),
    "Zandvoort":
        ("deciduous", "#1a301a", 24, "#10100a",
         [("water", "top"), ("dunes", 3), ("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Monza":
        ("deciduous", "#1a3a1a", 14, "#0a120a",
         [("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Baku City Circuit":
        (None, None, 0, "#0a0a12",
         [("water", "left"), ("buildings", "right", 7), ("grandstand", 0.0, 1)]),
    "Marina Bay":
        ("palm", "#1a4a1a", 25, "#0a0a12",
         [("buildings", "top", 8), ("water", "bottom"), ("grandstand", 0.0, 1)]),
    "COTA":
        (None, None, 0, "#100e08",
         [("tower", 0.50, 0.04), ("cactus_scatter",), ("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Autódromo Hermanos Rodríguez":
        ("deciduous", "#1a3a1a", 22, "#0a100a",
         [("stadium", 0.7, 1), ("grandstand", 0.0, 1)]),
    "Interlagos":
        ("deciduous", "#1a4a1a", 16, "#0a120a",
         [("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Las Vegas Strip":
        (None, None, 0, "#0a0a14",
         [("strip",), ("sphere", 0.90, 0.12), ("grandstand", 0.0, 1)]),
    "Lusail":
        (None, None, 0, "#120f08",
         [("dunes", 3), ("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Yas Marina":
        ("palm", "#1a4a1a", 25, "#0a0a10",
         [("water", "bottom"), ("buildings", "right", 4), ("grandstand", 0.0, 1)]),
    "Circuit":
        ("deciduous", "#1a3a1a", 22, "#0a100a",
         [("grandstand", 0.0, 1)]),
}

class ApexAI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ApexAI")
        self.root.configure(bg=BG)
        self.root.minsize(960, 780)
        self.root.geometry("1120x880")
        self.result = None
        self._logos = {}
        self._tk_images = []
        self._current_view = "predictions"
        self._schedule = []
        self._race_idx = -1
        self._season_driver_pts = {}
        self._season_team_pts = {}
        self._build()

    # -- Logo cache --
    def _logo(self, team: str, sz: int = 24):
        k = f"{team}_{sz}"
        if k not in self._logos:
            if HAS_PIL:
                img = load_logo(team, sz)
                if img:
                    if img.mode == "RGBA":
                        bg_rgb = tuple(int(BG_CARD.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                        solid = Image.new("RGBA", img.size, (*bg_rgb, 255))
                        solid.paste(img, (0, 0), img)
                        img = solid
                    self._logos[k] = ImageTk.PhotoImage(img)
                else:
                    self._logos[k] = None
            else:
                self._logos[k] = None
        return self._logos.get(k)

    # -- Build UI --
    def _build(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG, padx=36, pady=20)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Apex", font=("Helvetica Neue", 28, "bold"), fg=GOLD, bg=BG).pack(side=tk.LEFT)
        tk.Label(hdr, text="AI", font=("Helvetica Neue", 28, "bold"), fg=WHITE, bg=BG).pack(side=tk.LEFT)
        tk.Label(hdr, text="F1 Race Predictor", font=("Helvetica Neue", 12), fg=MUTED, bg=BG).pack(side=tk.LEFT, padx=(16, 0), pady=(8, 0))

        tk.Frame(self.root, bg=RED, height=2).pack(fill=tk.X, padx=36)

        # Controls
        ctrl = tk.Frame(self.root, bg=BG, padx=36, pady=16)
        ctrl.pack(fill=tk.X)

        self.btn_predict = self._make_btn(ctrl, "Predict Next Race", "#141415", WHITE, self._on_predict, border=BORDER)
        self.btn_predict.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_all = self._make_btn(ctrl, "Backtest All Races", "#141415", WHITE, self._on_all_races, border=BORDER)
        self.btn_all.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_viz = self._make_btn(ctrl, "Race Visualization", "#141415", WHITE, self._on_show_viz, border=BORDER)
        self.btn_viz.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_radio = self._make_btn(ctrl, "Team Radio", "#141415", WHITE, self._on_show_radio, border=BORDER)
        self.btn_radio.pack(side=tk.LEFT, padx=(0, 16))

        self._all_btns = [self.btn_predict, self.btn_all, self.btn_viz, self.btn_radio]

        self.status_lbl = tk.Label(ctrl, text="", font=("Helvetica Neue", 11), fg=MUTED, bg=BG)
        self.status_lbl.pack(side=tk.LEFT, padx=(8, 0))

        # Body – two panels (predictions view)
        self.body = tk.Frame(self.root, bg=BG, padx=36, pady=8)
        self.body.pack(fill=tk.BOTH, expand=True)
        body = self.body

        # Track visualization view (hidden initially)
        self.viz_frame = tk.Frame(self.root, bg=BG, padx=36, pady=8)

        # Team radio view (hidden initially)
        self.radio_frame = tk.Frame(self.root, bg=BG, padx=36, pady=8)
        self._radio_clips = []
        self._radio_playing = None
        self._radio_proc = None
        self._radio_wave_items = []
        self._radio_anim_running = False

        # Left panel – results
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        self.canvas = tk.Canvas(left, bg=BG, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(left, command=self.canvas.yview)
        self.results = tk.Frame(self.canvas, bg=BG)
        self.results.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.results, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vscroll.set)
        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for w in (self.canvas, self.results):
            w.bind("<MouseWheel>", self._scroll)

        # Right panel – algorithm (carbon fiber accent)
        right = tk.Frame(body, bg=BG_CARD, width=320, padx=20, pady=20)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        right.configure(highlightbackground=BORDER, highlightthickness=1)

        # Red accent stripe at top
        tk.Frame(right, bg=RED, height=2).pack(fill=tk.X, pady=(0, 12))
        tk.Label(right, text="ALGORITHM", font=("Helvetica Neue", 10, "bold"), fg=RED, bg=BG_CARD).pack(anchor="w")
        tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X, pady=(8, 12))

        code = tk.Text(right, font=("Menlo", 9), fg=GRAY, bg=BG_SURFACE, relief=tk.FLAT, padx=10, pady=10, wrap=tk.WORD, height=20, insertbackground=GRAY)
        code.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        code.insert("1.0", ALGO_TEXT)
        code.configure(state=tk.DISABLED)

        tk.Label(right, text="FEATURE IMPORTANCE", font=("Helvetica Neue", 10, "bold"), fg=GOLD, bg=BG_CARD).pack(anchor="w", pady=(8, 6))
        self.chart_lbl = tk.Label(right, bg=BG_CARD)
        self.chart_lbl.pack(anchor="w")

        # Initial state
        self._show_empty("Ready to predict.\nClick a button above to start.")

    def _make_btn(self, parent, text, bg_c, fg_c, cmd, border=None):
        frame = tk.Frame(parent, bg=bg_c, cursor="hand2",
                         highlightbackground=border or bg_c, highlightthickness=1)
        label = tk.Label(frame, text=text, font=("Helvetica Neue", 11, "bold"),
                         fg=fg_c, bg=bg_c, padx=18, pady=8, cursor="hand2")
        label.pack()
        label.bind("<Button-1>", lambda e: cmd())
        frame.bind("<Button-1>", lambda e: cmd())
        frame._label = label
        frame._default_bg = bg_c
        frame._default_fg = fg_c
        frame._current_bg = bg_c
        frame.bind("<Enter>", lambda e, f=frame: self._btn_hover(f, True))
        frame.bind("<Leave>", lambda e, f=frame: self._btn_hover(f, False))
        label.bind("<Enter>", lambda e, f=frame: self._btn_hover(f, True))
        label.bind("<Leave>", lambda e, f=frame: self._btn_hover(f, False))
        return frame

    def _btn_hover(self, btn, entering):
        c = btn._current_bg
        if entering:
            hover = self._lighten(c, 30)
            btn.configure(bg=hover, highlightbackground=hover)
            btn._label.configure(bg=hover)
        else:
            btn.configure(bg=c, highlightbackground=c)
            btn._label.configure(bg=c)

    def _set_active_btn(self, active_btn):
        for btn in self._all_btns:
            btn._current_bg = btn._default_bg
            btn.configure(bg=btn._default_bg,
                          highlightbackground=btn._default_bg)
            btn._label.configure(fg=btn._default_fg, bg=btn._default_bg)
        if active_btn:
            active_btn._current_bg = GOLD_DIM
            active_btn.configure(bg=GOLD_DIM, highlightbackground=GOLD_DIM)
            active_btn._label.configure(fg=GOLD, bg=GOLD_DIM)

    @staticmethod
    def _lighten(hex_c, amt=20):
        h = hex_c.lstrip("#")
        r, g, b = (min(255, int(h[i:i+2], 16) + amt) for i in (0, 2, 4))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _scroll(self, event):
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass
        return "break"

    # -- Status --
    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_lbl.configure(text=msg))

    def _set_busy(self, busy):
        pass

    # -- Predict next race --
    def _on_predict(self):
        self._set_active_btn(self.btn_predict)

        if self.result and self._schedule and "_model" in self.result:
            self._advance_and_predict()
            return

        self._set_status("Loading data...")
        self._clear()
        self._show_empty("Running predictions...")

        def work():
            r = run_predictions(progress_callback=self._set_status)
            self.root.after(0, lambda: self._show_predictions(r))

        threading.Thread(target=work, daemon=True).start()

    def _advance_and_predict(self):
        """Award points from current race prediction, advance to next race, re-predict."""
        r = self.result
        nr = r["next_race"]

        for i, p in enumerate(nr["predictions"]):
            pos = i + 1
            pts = F1_POINTS.get(pos, 0)
            if pts > 0:
                abbr = p["abbreviation"]
                team = p["team"]
                self._season_driver_pts[abbr] = self._season_driver_pts.get(abbr, 0) + pts
                self._season_team_pts[team] = self._season_team_pts.get(team, 0) + pts

        self._race_idx = (self._race_idx + 1) % len(self._schedule)
        race = self._schedule[self._race_idx]

        model = r["_model"]
        features = r["_features"]
        base_lineup = r["_base_lineup"]

        base_driver_pts = dict(r.get("_base_driver_pts", {}))
        base_team_pts = dict(r.get("_base_team_pts", {}))

        driver_pts = {k: base_driver_pts.get(k, 0) + self._season_driver_pts.get(k, 0)
                      for k in set(base_driver_pts) | set(self._season_driver_pts)}
        team_pts = {k: base_team_pts.get(k, 0) + self._season_team_pts.get(k, 0)
                    for k in set(base_team_pts) | set(self._season_team_pts)}

        extra = r.get("_extra_features", {})
        new_preds = predict_with_standings(model, features, base_lineup, driver_pts, team_pts,
                                           extra_features=extra)

        new_nr = {
            "year": nr["year"],
            "round": race["round"],
            "name": race["name"],
            "predicted_winner": new_preds[0]["abbreviation"],
            "top_probability": new_preds[0]["probability"],
            "predictions": new_preds,
        }
        r["next_race"] = new_nr

        self._set_status(f"Accuracy {r['accuracy']:.1%}  ·  Race {self._race_idx + 1}/{len(self._schedule)}")
        self._render_chart(r.get("feature_importance", {}))
        self._clear()
        self._display_prediction_ui(r)

    def _show_predictions(self, r):
        self._set_active_btn(self.btn_predict)
        if "error" in r:
            self._set_status(f"Error: {r['error'][:80]}")
            self._show_empty(f"Error\n\n{r['error'][:300]}")
            return

        self.result = r
        self._schedule = r.get("schedule", [])
        self._season_driver_pts = {}
        self._season_team_pts = {}

        nr = r["next_race"]
        self._race_idx = next(
            (i for i, s in enumerate(self._schedule) if s["round"] == nr["round"]),
            0
        )
        self._set_status(f"Accuracy {r['accuracy']:.1%}  ·  Race {self._race_idx + 1}/{len(self._schedule)}")
        self._render_chart(r.get("feature_importance", {}))
        self._clear()
        self._display_prediction_ui(r)

    def _display_prediction_ui(self, r):
        nr = r["next_race"]
        w = nr["predictions"][0]

        # -- Winner banner --
        banner = tk.Frame(self.results, bg=BG_CARD, padx=24, pady=20)
        banner.configure(highlightbackground=BORDER, highlightthickness=1)
        banner.pack(fill=tk.X, pady=(0, 16))

        tk.Label(banner, text=f"NEXT RACE · {nr['name']} ({nr['year']})", font=("Helvetica Neue", 10, "bold"), fg=GOLD, bg=BG_CARD).pack(anchor="w", pady=(0, 12))

        wb = tk.Frame(banner, bg=GOLD_DIM, padx=20, pady=16)
        wb.pack(fill=tk.X, pady=(0, 16))
        logo = self._logo(w["team"], 44)
        if logo:
            tk.Label(wb, image=logo, bg=GOLD_DIM).pack(side=tk.LEFT, padx=(0, 16))
        wt = tk.Frame(wb, bg=GOLD_DIM)
        wt.pack(side=tk.LEFT)
        tk.Label(wt, text="PREDICTED WINNER", font=("Helvetica Neue", 9, "bold"), fg=GOLD_GLOW, bg=GOLD_DIM).pack(anchor="w")
        tk.Label(wt, text=w["abbreviation"], font=("Helvetica Neue", 32, "bold"), fg=WHITE, bg=GOLD_DIM).pack(anchor="w")
        tk.Label(wt, text=f"{w['probability']*100:.1f}% win probability  ·  {w['team']}", font=("Helvetica Neue", 11), fg=GRAY, bg=GOLD_DIM).pack(anchor="w")

        # -- Podium --
        pod = tk.Frame(banner, bg=BG_CARD)
        pod.pack(fill=tk.X, pady=(0, 16))
        heights = [60, 80, 50]
        for col_idx, (rank_idx, h) in enumerate(zip([1, 0, 2], heights)):
            p = nr["predictions"][rank_idx]
            c = tk.Frame(pod, bg=BG_CARD)
            c.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

            pl = self._logo(p["team"], 24)
            if pl:
                tk.Label(c, image=pl, bg=BG_CARD).pack(pady=(0, 4))

            sz = 20 if rank_idx == 0 else 15
            tk.Label(c, text=p["abbreviation"], font=("Helvetica Neue", sz, "bold"), fg=tc(p["team"]), bg=BG_CARD).pack()
            tk.Label(c, text=f"{p['probability']*100:.1f}%", font=("Helvetica Neue", 10), fg=GRAY, bg=BG_CARD).pack()

            pedestal = tk.Frame(c, bg=tc(p["team"]), height=h, width=80)
            pedestal.pack(side=tk.BOTTOM, pady=(4, 0))
            pedestal.pack_propagate(False)
            pos_text = ["1st", "2nd", "3rd"][rank_idx]
            tk.Label(pedestal, text=pos_text, font=("Helvetica Neue", 11, "bold"), fg=WHITE, bg=tc(p["team"])).pack(expand=True)

        # -- Full grid --
        grid_card = tk.Frame(self.results, bg=BG_CARD, padx=20, pady=16)
        grid_card.configure(highlightbackground=BORDER, highlightthickness=1)
        grid_card.pack(fill=tk.X, pady=(0, 16))
        tk.Label(grid_card, text="FULL GRID", font=("Helvetica Neue", 10, "bold"), fg=GOLD, bg=BG_CARD).pack(anchor="w", pady=(0, 10))

        for i, p in enumerate(nr["predictions"]):
            self._driver_row(grid_card, i + 1, p["abbreviation"], p["team"], p["probability"], highlight=(i == 0))

        # -- Last race --
        lr = r["last_race"]
        lr_card = tk.Frame(self.results, bg=BG_CARD, padx=20, pady=16)
        lr_card.configure(highlightbackground=BORDER, highlightthickness=1)
        lr_card.pack(fill=tk.X, pady=(0, 16))
        tk.Label(lr_card, text=f"LAST RACE · {lr['name']} ({lr['year']})", font=("Helvetica Neue", 10, "bold"), fg=GOLD, bg=BG_CARD).pack(anchor="w", pady=(0, 10))

        pred, actual = lr["predicted_winner"], lr["actual_winner"]
        hit = pred == actual
        tk.Label(lr_card, text=f"Predicted: {pred}  ·  Actual: {actual}  {'✓' if hit else '✗'}", font=("Helvetica Neue", 11, "bold"), fg=GREEN if hit else RED, bg=BG_CARD).pack(anchor="w", pady=(0, 10))

        for i, p in enumerate(lr["predictions"][:10]):
            self._driver_row(lr_card, i + 1, p["abbreviation"], p["team"], p["probability"], highlight=(p["abbreviation"] == actual))

        # Accuracy footer
        tk.Label(self.results, text=f"Model accuracy: {r['accuracy']:.1%}", font=("Helvetica Neue", 10), fg=MUTED, bg=BG).pack(anchor="w", pady=(8, 20))

    # -- Backtest all races --
    def _on_all_races(self):
        self.result = None
        self._schedule = []
        self._race_idx = -1
        self._season_driver_pts = {}
        self._season_team_pts = {}
        self._set_active_btn(self.btn_all)
        self._set_status("Backtesting every race...")
        self._clear()
        self._show_empty("Running backtest on every race...\nThis takes a few minutes.")

        def work():
            r = run_predictions_all_races(progress_callback=self._set_status)
            self.root.after(0, lambda: self._show_all_races(r))

        threading.Thread(target=work, daemon=True).start()

    def _show_all_races(self, r):
        self._set_active_btn(self.btn_all)
        if "error" in r:
            self._set_status(f"Error: {r['error'][:80]}")
            self._show_empty(f"Error\n\n{r['error'][:300]}")
            return

        self._set_status(f"Backtest: {r['correct']}/{r['total']} ({r['accuracy']:.1%})")
        self._clear()

        card = tk.Frame(self.results, bg=BG_CARD, padx=20, pady=16)
        card.configure(highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 16))

        tk.Label(card, text=f"BACKTEST · {r['correct']}/{r['total']} correct ({r['accuracy']:.1%})", font=("Helvetica Neue", 11, "bold"), fg=GOLD, bg=BG_CARD).pack(anchor="w", pady=(0, 12))

        # Header
        hdr = tk.Frame(card, bg=BG_SURFACE, padx=10, pady=6)
        hdr.pack(fill=tk.X, pady=(0, 4))
        for txt, w in [("Year", 5), ("Race", 22), ("Predicted", 10), ("Actual", 10), ("", 3)]:
            tk.Label(hdr, text=txt, font=("Helvetica Neue", 9, "bold"), fg=MUTED, bg=BG_SURFACE, width=w, anchor="w").pack(side=tk.LEFT, padx=2)

        for race in r["all_races"]:
            row = tk.Frame(card, bg=BG_CARD, padx=10, pady=4)
            row.pack(fill=tk.X)
            fg = GREEN if race["correct"] else RED
            mark = "✓" if race["correct"] else "✗"
            race_name = race.get("name", f"R{race['round']}")
            for txt, w in [(str(race["year"]), 5), (race_name, 22), (race["predicted"], 10), (race["actual"], 10), (mark, 3)]:
                tk.Label(row, text=txt, font=("Menlo", 10), fg=fg if txt == mark else GRAY, bg=BG_CARD, width=w, anchor="w").pack(side=tk.LEFT, padx=2)

    # -- Shared helpers --
    def _driver_row(self, parent, rank, abbr, team, prob, highlight=False):
        color = tc(team)
        fg = WHITE if highlight else GRAY
        bg = BG_SURFACE if highlight else BG_CARD

        row = tk.Frame(parent, bg=bg, padx=8, pady=5)
        row.pack(fill=tk.X, pady=1)

        tk.Label(row, text=f"P{rank}", font=("Helvetica Neue", 10, "bold"), fg=MUTED, bg=bg, width=3).pack(side=tk.LEFT, padx=(0, 6))

        # Color bar
        bar = tk.Frame(row, width=3, height=18, bg=color)
        bar.pack(side=tk.LEFT, padx=(0, 8))
        bar.pack_propagate(False)

        logo = self._logo(team, 20)
        if logo:
            tk.Label(row, image=logo, bg=bg).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(row, text=abbr, font=("Helvetica Neue", 12, "bold"), fg=fg, bg=bg).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text=team, font=("Helvetica Neue", 9), fg=MUTED, bg=bg).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Probability with inline bar
        pf = tk.Frame(row, bg=bg, width=130, height=18)
        pf.pack(side=tk.RIGHT)
        pf.pack_propagate(False)
        bw = max(2, int(80 * min(prob, 1)))
        tk.Frame(pf, bg=BORDER, height=4, width=80).place(x=0, y=7)
        tk.Frame(pf, bg=color, height=4, width=bw).place(x=0, y=7)
        tk.Label(pf, text=f"{prob*100:.1f}%", font=("Helvetica Neue", 10, "bold"), fg=fg, bg=bg).place(x=86, y=0)

    def _render_chart(self, fi):
        self.chart_lbl.configure(image="")
        if not fi:
            return
        fig, ax = plt.subplots(figsize=(3.6, 2.4), facecolor=BG_CARD)
        ax.set_facecolor(BG_CARD)
        labels = list(fi.keys())
        vals = list(fi.values())
        colors = [GOLD if i == 0 else GOLD_DIM for i in range(len(labels))]
        ax.barh(range(len(labels)), vals, color=colors, height=0.5, edgecolor=BORDER)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8, color=GRAY)
        ax.tick_params(axis="x", colors=MUTED, labelsize=7)
        ax.tick_params(axis="y", left=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
        ax.set_xlim(0, max(vals) * 1.15)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=BG_CARD)
        plt.close()
        buf.seek(0)
        if HAS_PIL:
            img = Image.open(buf).convert("RGB")
            photo = ImageTk.PhotoImage(img)
            self.chart_lbl.configure(image=photo)
            self.chart_lbl.image = photo

    def _clear(self):
        for w in self.results.winfo_children():
            w.destroy()

    def _show_empty(self, text):
        self._clear()
        tk.Label(self.results, text=text, font=("Helvetica Neue", 13), fg=MUTED, bg=BG, justify=tk.CENTER).pack(expand=True, pady=80)

    # ── View switching ──

    def _switch_to_view(self, view):
        self._current_view = view
        self._anim_running = False
        self.body.pack_forget()
        self.viz_frame.pack_forget()
        self.radio_frame.pack_forget()
        if view == "predictions":
            self.body.pack(fill=tk.BOTH, expand=True)
            self._set_active_btn(None)
        elif view == "viz":
            self.viz_frame.pack(fill=tk.BOTH, expand=True)
            self._set_active_btn(self.btn_viz)
        elif view == "radio":
            self.radio_frame.pack(fill=tk.BOTH, expand=True)
            self._set_active_btn(self.btn_radio)

    def _on_show_viz(self):
        if self._current_view == "viz":
            self._switch_to_view("predictions")
            return
        if not self.result:
            self._set_status("Run a prediction first")
            return
        self._build_viz()
        self._switch_to_view("viz")

    def _on_show_radio(self):
        if self._current_view == "radio":
            self._stop_radio()
            self._switch_to_view("predictions")
            return
        if not HAS_RADIO:
            self._set_status("f1radio package not installed (pip install f1radio)")
            return
        self._build_radio()
        self._switch_to_view("radio")

    # ── Team Radio ──

    RADIO_RACES = [
        (2025, "Australia"), (2025, "China"), (2025, "Japan"),
        (2025, "Bahrain"), (2025, "Saudi Arabia"), (2025, "Miami"),
        (2025, "Emilia Romagna"), (2025, "Monaco"), (2025, "Spain"),
        (2025, "Canada"), (2025, "Austria"), (2025, "Great Britain"),
        (2025, "Belgium"), (2025, "Hungary"), (2025, "Netherlands"),
        (2025, "Italy"), (2025, "Azerbaijan"), (2025, "Singapore"),
        (2025, "United States"), (2025, "Mexico"), (2025, "Brazil"),
        (2025, "Las Vegas"), (2025, "Qatar"), (2025, "Abu Dhabi"),
    ]

    def _build_radio(self):
        self._stop_radio()
        for w in self.radio_frame.winfo_children():
            w.destroy()
        self._radio_tk_images = []

        # Header
        top = tk.Frame(self.radio_frame, bg=BG)
        top.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top, text="TEAM RADIO", font=("Helvetica Neue", 16, "bold"),
                 fg=GOLD, bg=BG).pack(side=tk.LEFT)
        tk.Label(top, text="Listen to driver-team communications",
                 font=("Helvetica Neue", 11), fg=MUTED, bg=BG
                 ).pack(side=tk.LEFT, padx=(16, 0), pady=(3, 0))

        tk.Frame(self.radio_frame, bg=BORDER, height=1).pack(fill=tk.X)

        # Two-column layout
        content = tk.Frame(self.radio_frame, bg=BG)
        content.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Left: race selector
        left_panel = tk.Frame(content, bg=BG_CARD, width=220, padx=12, pady=12)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left_panel.pack_propagate(False)
        left_panel.configure(highlightbackground=BORDER, highlightthickness=1)

        tk.Label(left_panel, text="SELECT RACE", font=("Helvetica Neue", 9, "bold"),
                 fg=GOLD, bg=BG_CARD).pack(anchor="w")
        tk.Frame(left_panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(6, 8))

        race_scroll_frame = tk.Frame(left_panel, bg=BG_CARD)
        race_scroll_frame.pack(fill=tk.BOTH, expand=True)

        race_canvas = tk.Canvas(race_scroll_frame, bg=BG_CARD, highlightthickness=0)
        race_sb = ttk.Scrollbar(race_scroll_frame, orient="vertical", command=race_canvas.yview)
        race_inner = tk.Frame(race_canvas, bg=BG_CARD)
        race_inner.bind("<Configure>", lambda e: race_canvas.configure(scrollregion=race_canvas.bbox("all")))
        race_canvas.create_window((0, 0), window=race_inner, anchor="nw")
        race_canvas.configure(yscrollcommand=race_sb.set)
        race_sb.pack(side=tk.RIGHT, fill=tk.Y)
        race_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for w in (race_canvas, race_inner):
            w.bind("<MouseWheel>", lambda e, c=race_canvas: (
                c.yview_scroll(int(-1 * (e.delta / 120)), "units"), "break"
            )[-1])

        self._race_btns = []
        for yr, race in self.RADIO_RACES:
            rb = tk.Frame(race_inner, bg=BG_SURFACE, cursor="hand2",
                          highlightbackground=BORDER, highlightthickness=1)
            rb.pack(fill=tk.X, pady=2)
            lbl = tk.Label(rb, text=f"  {race}", font=("Helvetica Neue", 10),
                           fg=GRAY, bg=BG_SURFACE, anchor="w", padx=6, pady=5,
                           cursor="hand2")
            lbl.pack(fill=tk.X)
            rb._lbl = lbl
            rb._race = (yr, race)
            for w in (rb, lbl):
                w.bind("<Button-1>", lambda e, r=(yr, race), b=rb: self._load_radio_race(r, b))
                w.bind("<Enter>", lambda e, b=rb: (
                    b.configure(bg=BG_HOVER), b._lbl.configure(bg=BG_HOVER)
                ))
                w.bind("<Leave>", lambda e, b=rb: (
                    b.configure(bg=getattr(b, '_sel_bg', BG_SURFACE)),
                    b._lbl.configure(bg=getattr(b, '_sel_bg', BG_SURFACE))
                ))
            self._race_btns.append(rb)

        # Right: clips panel
        right_panel = tk.Frame(content, bg=BG_CARD, padx=16, pady=16)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_panel.configure(highlightbackground=BORDER, highlightthickness=1)

        # Now-playing bar
        np_bar = tk.Frame(right_panel, bg=BG_SURFACE, padx=12, pady=10)
        np_bar.pack(fill=tk.X, pady=(0, 10))
        np_bar.configure(highlightbackground=BORDER, highlightthickness=1)

        self._np_canvas = tk.Canvas(np_bar, bg=BG_SURFACE, highlightthickness=0,
                                     height=36, width=50)
        self._np_canvas.pack(side=tk.LEFT, padx=(0, 10))

        np_text = tk.Frame(np_bar, bg=BG_SURFACE)
        np_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._np_driver = tk.Label(np_text, text="No clip playing",
                                    font=("Helvetica Neue", 12, "bold"),
                                    fg=WHITE, bg=BG_SURFACE, anchor="w")
        self._np_driver.pack(anchor="w")
        self._np_detail = tk.Label(np_text, text="Select a race to load team radio",
                                    font=("Helvetica Neue", 9), fg=MUTED,
                                    bg=BG_SURFACE, anchor="w")
        self._np_detail.pack(anchor="w")

        stop_btn = self._make_btn(np_bar, "Stop", "#3a1a1a", RED,
                                   self._stop_radio, border="#4a2222")
        stop_btn.pack(side=tk.RIGHT)

        # Clip list header
        hdr = tk.Frame(right_panel, bg=BG_CARD)
        hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(hdr, text="RADIO CLIPS", font=("Helvetica Neue", 9, "bold"),
                 fg=GOLD, bg=BG_CARD).pack(side=tk.LEFT)
        self._clip_count_lbl = tk.Label(hdr, text="", font=("Helvetica Neue", 9),
                                         fg=MUTED, bg=BG_CARD)
        self._clip_count_lbl.pack(side=tk.RIGHT)

        tk.Frame(right_panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 6))

        # Scrollable clip list
        clip_scroll = tk.Frame(right_panel, bg=BG_CARD)
        clip_scroll.pack(fill=tk.BOTH, expand=True)

        self._clip_canvas = tk.Canvas(clip_scroll, bg=BG_CARD, highlightthickness=0)
        clip_sb = ttk.Scrollbar(clip_scroll, orient="vertical", command=self._clip_canvas.yview)
        self._clip_inner = tk.Frame(self._clip_canvas, bg=BG_CARD)
        self._clip_inner.bind("<Configure>",
                               lambda e: self._clip_canvas.configure(
                                   scrollregion=self._clip_canvas.bbox("all")))
        self._clip_canvas.create_window((0, 0), window=self._clip_inner, anchor="nw")
        self._clip_canvas.configure(yscrollcommand=clip_sb.set)
        clip_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._clip_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for w in (self._clip_canvas, self._clip_inner):
            w.bind("<MouseWheel>", lambda e: (
                self._clip_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
                "break"
            )[-1])

        self._radio_loading_lbl = tk.Label(self._clip_inner, text="Select a race from the left panel",
                                            font=("Helvetica Neue", 12), fg=MUTED, bg=BG_CARD,
                                            pady=40)
        self._radio_loading_lbl.pack()

    def _load_radio_race(self, race_info, btn):
        yr, race_name = race_info

        for rb in self._race_btns:
            rb._sel_bg = BG_SURFACE
            rb.configure(bg=BG_SURFACE)
            rb._lbl.configure(bg=BG_SURFACE, fg=GRAY)
        btn._sel_bg = GOLD_DIM
        btn.configure(bg=GOLD_DIM)
        btn._lbl.configure(bg=GOLD_DIM, fg=GOLD)

        self._stop_radio()
        for w in self._clip_inner.winfo_children():
            w.destroy()
        self._radio_loading_lbl = tk.Label(self._clip_inner,
                                            text=f"Loading radio clips for {race_name}...",
                                            font=("Helvetica Neue", 12), fg=GOLD, bg=BG_CARD,
                                            pady=40)
        self._radio_loading_lbl.pack()
        self._clip_count_lbl.configure(text="loading...")

        def fetch():
            try:
                session = f1radio.load(yr, race_name, "R")
                clips = session.clips
                self.root.after(0, lambda: self._populate_clips(clips, race_name))
            except Exception as e:
                self.root.after(0, lambda: self._radio_error(str(e)))

        threading.Thread(target=fetch, daemon=True).start()

    def _radio_error(self, msg):
        for w in self._clip_inner.winfo_children():
            w.destroy()
        tk.Label(self._clip_inner, text=f"Error: {msg}",
                 font=("Helvetica Neue", 11), fg=RED, bg=BG_CARD,
                 wraplength=400, pady=30).pack()
        self._clip_count_lbl.configure(text="error")

    def _populate_clips(self, clips, race_name):
        for w in self._clip_inner.winfo_children():
            w.destroy()
        self._radio_clips = clips
        self._clip_count_lbl.configure(text=f"{len(clips)} clips")

        if not clips:
            tk.Label(self._clip_inner, text="No radio clips available for this race.",
                     font=("Helvetica Neue", 12), fg=MUTED, bg=BG_CARD, pady=30).pack()
            return

        for i, clip in enumerate(clips):
            self._make_clip_row(i, clip)

    def _make_clip_row(self, idx, clip):
        color = tc(clip.team) if clip.team else GRAY
        bg = BG_SURFACE if idx % 2 == 0 else BG_CARD

        row = tk.Frame(self._clip_inner, bg=bg, padx=10, pady=8, cursor="hand2")
        row.pack(fill=tk.X, pady=1)

        # Color accent bar
        accent = tk.Frame(row, bg=color, width=4)
        accent.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Play indicator
        play_lbl = tk.Label(row, text="\u25B6", font=("Helvetica Neue", 14),
                            fg=color, bg=bg, cursor="hand2")
        play_lbl.pack(side=tk.LEFT, padx=(0, 10))

        # Driver info
        info = tk.Frame(row, bg=bg)
        info.pack(side=tk.LEFT, fill=tk.X, expand=True)

        driver_text = clip.driver_name or clip.driver or "Unknown"
        team_text = clip.team or ""
        tk.Label(info, text=driver_text,
                 font=("Helvetica Neue", 11, "bold"), fg=WHITE, bg=bg,
                 anchor="w").pack(anchor="w")

        detail_parts = [team_text]
        if clip.context and getattr(clip.context, "position", None):
            detail_parts.append(f"P{clip.context.position}")
        if clip.context and getattr(clip.context, "compound", None):
            detail_parts.append(clip.context.compound)
        if clip.context and getattr(clip.context, "lap_number", None):
            detail_parts.append(f"Lap {clip.context.lap_number}")
        detail = "  |  ".join(p for p in detail_parts if p)

        tk.Label(info, text=detail, font=("Helvetica Neue", 9),
                 fg=MUTED, bg=bg, anchor="w").pack(anchor="w")

        # Clip number
        tk.Label(row, text=f"#{idx + 1}", font=("Helvetica Neue", 9),
                 fg=MUTED, bg=bg).pack(side=tk.RIGHT, padx=(10, 0))

        row._clip_idx = idx
        row._play_lbl = play_lbl
        row._orig_bg = bg
        for w in (row, play_lbl) + tuple(info.winfo_children()) + (info, accent):
            w.bind("<Button-1>", lambda e, ci=idx: self._play_radio_clip(ci))
            w.bind("<Enter>", lambda e, r=row: self._clip_hover(r, True))
            w.bind("<Leave>", lambda e, r=row: self._clip_hover(r, False))

    def _clip_hover(self, row, entering):
        bg = BG_HOVER if entering else row._orig_bg
        row.configure(bg=bg)
        for w in row.winfo_children():
            if isinstance(w, (tk.Label, tk.Frame)):
                try:
                    w.configure(bg=bg)
                    for c in w.winfo_children():
                        if isinstance(c, tk.Label):
                            c.configure(bg=bg)
                except Exception:
                    pass

    def _play_radio_clip(self, idx):
        if idx >= len(self._radio_clips):
            return
        clip = self._radio_clips[idx]

        self._stop_radio()

        path = clip.local_path
        if not path or not os.path.exists(path):
            self._set_status("Audio file not available")
            return

        self._radio_playing = idx
        driver_text = clip.driver_name or clip.driver or "Unknown"
        team_text = clip.team or ""
        self._np_driver.configure(text=f"{driver_text}", fg=tc(team_text) if team_text else WHITE)
        self._np_detail.configure(text=f"{team_text}  |  Clip #{idx + 1}")

        try:
            self._radio_proc = subprocess.Popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            try:
                self._radio_proc = subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except FileNotFoundError:
                self._set_status("No audio player found (need afplay or ffplay)")
                return

        self._radio_anim_running = True
        self._radio_wave_frame = 0
        self._radio_wave_tick()

        def wait_for_end():
            self._radio_proc.wait()
            self.root.after(0, self._on_radio_ended)
        threading.Thread(target=wait_for_end, daemon=True).start()

    def _stop_radio(self):
        self._radio_anim_running = False
        if self._radio_proc:
            try:
                self._radio_proc.terminate()
                self._radio_proc.wait(timeout=2)
            except Exception:
                try:
                    self._radio_proc.kill()
                except Exception:
                    pass
            self._radio_proc = None
        self._radio_playing = None
        if hasattr(self, "_np_driver"):
            self._np_driver.configure(text="No clip playing", fg=WHITE)
            self._np_detail.configure(text="Select a clip to play")
        if hasattr(self, "_np_canvas"):
            self._np_canvas.delete("all")

    def _on_radio_ended(self):
        self._radio_anim_running = False
        self._radio_playing = None
        if hasattr(self, "_np_canvas"):
            self._np_canvas.delete("all")
        if hasattr(self, "_np_driver"):
            self._np_driver.configure(text="No clip playing", fg=WHITE)
            self._np_detail.configure(text="Playback finished")

    def _radio_wave_tick(self):
        if not self._radio_anim_running or self._current_view != "radio":
            return
        c = self._np_canvas
        c.delete("all")
        w = 50
        h = 36
        frame = self._radio_wave_frame
        bars = 8
        bar_w = w / bars
        for i in range(bars):
            amp = 0.3 + 0.7 * abs(math.sin(frame * 0.12 + i * 0.8))
            bh = amp * (h - 4)
            x0 = i * bar_w + 2
            x1 = x0 + bar_w - 2
            y0 = (h - bh) / 2
            y1 = y0 + bh
            color_val = int(180 + 75 * amp)
            green_val = int(160 * amp)
            bar_color = f"#{min(255, color_val):02x}{green_val:02x}20"
            c.create_rectangle(x0, y0, x1, y1, fill=bar_color, outline="")
        self._radio_wave_frame += 1
        self.root.after(50, self._radio_wave_tick)

    # ── Track visualization ──

    def _build_viz(self):
        self._anim_running = False
        for w in self.viz_frame.winfo_children():
            w.destroy()
        self._tk_images = []

        r = self.result
        nr = r["next_race"]
        preds = nr["predictions"]
        race_name = nr.get("name", "")

        track_pts, circuit_name = get_track(race_name)

        top = tk.Frame(self.viz_frame, bg=BG)
        top.pack(fill=tk.X, pady=(0, 8))
        self._make_btn(top, "← Back to Results", BG_CARD, GRAY,
                       lambda: self._switch_to_view("predictions"), border=BORDER).pack(side=tk.LEFT)
        tk.Label(top, text=f"RACE VISUALIZATION · {nr['name']} ({nr['year']})",
                 font=("Helvetica Neue", 14, "bold"), fg=GOLD, bg=BG).pack(side=tk.LEFT, padx=(20, 0))
        tk.Label(top, text=circuit_name,
                 font=("Helvetica Neue", 11), fg=MUTED, bg=BG).pack(side=tk.LEFT, padx=(12, 0))

        self.track_canvas = tk.Canvas(self.viz_frame, bg=BG, highlightthickness=0)
        self.track_canvas.pack(fill=tk.BOTH, expand=True)

        self._viz_preds = preds
        self._viz_raw_pts = track_pts
        self._viz_circuit = circuit_name
        self._viz_drawn = False
        self.track_canvas.bind("<Configure>", self._on_viz_resize)

    def _on_viz_resize(self, event=None):
        cw = self.track_canvas.winfo_width()
        ch = self.track_canvas.winfo_height()
        if cw < 200 or ch < 200:
            return
        self._anim_running = False
        self.track_canvas.delete("all")
        self._tk_images = []
        self._draw_real_track(cw, ch, self._viz_raw_pts, self._viz_preds)

    # ── Interpolation helpers ──

    @staticmethod
    def _catmull_rom(p0, p1, p2, p3, t):
        t2, t3 = t * t, t * t * t
        x = 0.5 * ((2 * p1[0]) +
                    (-p0[0] + p2[0]) * t +
                    (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                    (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
        y = 0.5 * ((2 * p1[1]) +
                    (-p0[1] + p2[1]) * t +
                    (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                    (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
        return (x, y)

    def _interpolate_track(self, raw_pts, num_out=400):
        n = len(raw_pts)
        pts = []
        seg_steps = max(2, num_out // n)
        for i in range(n):
            p0 = raw_pts[(i - 1) % n]
            p1 = raw_pts[i]
            p2 = raw_pts[(i + 1) % n]
            p3 = raw_pts[(i + 2) % n]
            for s in range(seg_steps):
                t = s / seg_steps
                pts.append(self._catmull_rom(p0, p1, p2, p3, t))
        return pts

    def _track_normal(self, pts, idx):
        p0 = pts[idx]
        p1 = pts[(idx + 1) % len(pts)]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.sqrt(dx * dx + dy * dy) or 1
        return (-dy / length, dx / length)

    # ── Smooth position helpers ──

    def _pos_at(self, t):
        """Get smooth (x, y) at float position t along self._track_pts."""
        track = self._track_pts
        num = len(track)
        t = t % num
        i0 = int(t) % num
        i1 = (i0 + 1) % num
        frac = t - int(t)
        return (track[i0][0] + (track[i1][0] - track[i0][0]) * frac,
                track[i0][1] + (track[i1][1] - track[i0][1]) * frac)

    def _normal_at(self, t):
        """Get smooth normal at float position t."""
        track = self._track_pts
        num = len(track)
        t = t % num
        i0 = int(t) % num
        i1 = (i0 + 1) % num
        dx = track[i1][0] - track[i0][0]
        dy = track[i1][1] - track[i0][1]
        length = math.sqrt(dx * dx + dy * dy) or 1
        return (-dy / length, dx / length)

    # ── Scene drawing ──

    def _draw_scene(self, canvas, cw, ch, track, hw, circuit_name):
        scene = SCENES.get(circuit_name, SCENES.get("Circuit"))
        if not scene:
            return
        veg_type, veg_color, veg_spacing, ground_color, features = scene
        s = min(cw, ch) / 800.0
        num = len(track)
        tw = int(hw * 2)

        flat = []
        for p in track:
            flat.extend(p)
        canvas.create_polygon(flat, outline=ground_color, fill="",
                              width=tw * 3, smooth=True)

        for feat in features:
            try:
                kind = feat[0]
                if kind == "grandstand":
                    continue
                elif kind == "water":
                    self._s_water_edge(canvas, cw, ch, feat[1], s)
                elif kind == "mountains":
                    self._s_mountain_range(canvas, cw, ch * 0.06, feat[1], s)
                elif kind == "dunes":
                    self._s_dune_field(canvas, cw, ch, feat[1], s)
                elif kind in ("buildings", "skyline"):
                    self._s_building_row(canvas, cw, ch, feat[1], feat[2], s)
                elif kind == "strip":
                    self._s_vegas_strip(canvas, cw, ch, s)
                elif kind == "sphere":
                    self._s_sphere(canvas, feat[1] * cw, feat[2] * ch, 25 * s)
                elif kind == "ferris":
                    self._s_ferris(canvas, feat[1] * cw, feat[2] * ch, 30 * s)
                elif kind == "lake":
                    xs = [p[0] for p in track]
                    ys = [p[1] for p in track]
                    cx = (min(xs) + max(xs)) / 2
                    cy = (min(ys) + max(ys)) / 2
                    rw = (max(xs) - min(xs)) * 0.2
                    rh = (max(ys) - min(ys)) * 0.15
                    self._s_lake(canvas, cx, cy - 15 * s, rw, rh)
                elif kind == "yachts":
                    self._s_yacht_harbor(canvas, cw, ch, feat[1], s)
                elif kind == "tower":
                    self._s_tower(canvas, feat[1] * cw, feat[2] * ch, s)
                elif kind == "stadium":
                    self._s_stadium(canvas, track, hw, feat[1], feat[2], s)
                elif kind == "cactus_scatter":
                    self._s_cactus_scatter(canvas, track, hw, s)
            except Exception:
                pass

        if veg_type and veg_spacing > 0:
            self._draw_tree_line(canvas, track, hw, veg_type, veg_spacing,
                                 veg_color or "#1a3a1a", s)

        for feat in features:
            if feat[0] == "grandstand":
                try:
                    self._s_grandstand_at(canvas, track, hw, feat[1], feat[2], s)
                except Exception:
                    pass

    def _s_tree(self, c, x, y, s, kind, color):
        tw = max(1, 2 * s)
        if kind == "deciduous":
            c.create_line(x, y, x, y - 14 * s, fill="#2a1a0a", width=tw)
            r = 10 * s
            c.create_oval(x - r, y - 14 * s - r * 1.2, x + r,
                          y - 14 * s + r * 0.4, fill=color, outline="")
        elif kind == "pine":
            c.create_line(x, y, x, y - 8 * s, fill="#2a1a0a", width=tw)
            for hw_v, off in [(4, 0), (6, 6), (8, 12)]:
                by = y - 8 * s - off * s
                c.create_polygon(x - hw_v * s, by, x, by - 10 * s,
                                 x + hw_v * s, by, fill=color, outline="")
        elif kind == "palm":
            c.create_line(x, y, x + 2 * s, y - 35 * s, x + 1 * s,
                          y - 45 * s, fill="#5c3a14",
                          width=max(2, 3 * s), smooth=True)
            tx, ty = x + 1 * s, y - 45 * s
            for dx, dy in [(-22, -8), (-16, -15), (-8, -18), (6, -17),
                           (14, -12), (20, -5), (-18, 2), (16, 4)]:
                c.create_line(tx, ty, tx + dx * s, ty + dy * s,
                              fill="#1a5a1a", width=max(1, 2 * s))
        elif kind == "cherry":
            c.create_line(x, y, x, y - 12 * s, fill="#3a2014", width=tw)
            r = 9 * s
            c.create_oval(x - r, y - 12 * s - r * 1.2, x + r,
                          y - 12 * s + r * 0.4, fill="#3a1a28", outline="")
            for dx, dy in [(-5, -3), (3, -6), (6, 0), (-3, 2), (0, -8)]:
                pr = 2 * s
                c.create_oval(x + dx * s - pr, y - 16 * s + dy * s - pr,
                              x + dx * s + pr, y - 16 * s + dy * s + pr,
                              fill="#ffb7c5", outline="")

    def _draw_tree_line(self, c, track, hw, kind, spacing, color, s):
        num = len(track)
        offset_base = hw * 2.5
        for i in range(0, num, spacing):
            nx, ny = self._track_normal(track, i)
            for side in (1, -1):
                if (i // spacing + (1 if side > 0 else 0)) % 5 == 0:
                    continue
                dist = offset_base + ((i * 7) % 11 - 5) * s
                tree_s = s * (0.6 + ((i * 13) % 7) * 0.08)
                tx = track[i][0] + nx * dist * side
                ty = track[i][1] + ny * dist * side
                self._s_tree(c, tx, ty, tree_s, kind, color)

    def _s_grandstand_at(self, c, track, hw, frac, side, s):
        num = len(track)
        base = int(frac * num) % num
        span = max(8, int(num * 0.06))
        for row in range(4):
            inner_d = hw * 2.4 + row * 5 * s
            outer_d = inner_d + 4 * s
            pts = []
            for j in range(-span // 2, span // 2 + 1, 2):
                idx = (base + j) % num
                nx, ny = self._track_normal(track, idx)
                pts.extend([track[idx][0] + nx * inner_d * side,
                            track[idx][1] + ny * inner_d * side])
            for j in range(span // 2, -span // 2 - 1, -2):
                idx = (base + j) % num
                nx, ny = self._track_normal(track, idx)
                pts.extend([track[idx][0] + nx * outer_d * side,
                            track[idx][1] + ny * outer_d * side])
            if len(pts) >= 6:
                gray = 0x14 + row * 4
                c.create_polygon(pts, fill=f"#{gray:02x}{gray:02x}{gray + 8:02x}",
                                 outline="#222230", smooth=True)

    def _s_water_edge(self, c, cw, ch, side, s):
        depth = 0.12
        col = "#0a1428"
        if side == "bottom":
            y0 = ch * (1 - depth)
            c.create_rectangle(0, y0, cw, ch, fill=col, outline="")
            for i, f in enumerate([0.3, 0.55, 0.8]):
                c.create_line(cw * 0.05, y0 + depth * ch * f,
                              cw * 0.95, y0 + depth * ch * f,
                              fill="#12203a", width=1, dash=(15 + i * 8, 25))
        elif side == "top":
            y1 = ch * depth
            c.create_rectangle(0, 0, cw, y1, fill=col, outline="")
            for f in [0.3, 0.55, 0.8]:
                c.create_line(cw * 0.05, y1 * f, cw * 0.95, y1 * f,
                              fill="#12203a", width=1, dash=(20, 25))
        elif side == "left":
            c.create_rectangle(0, 0, cw * depth, ch, fill=col, outline="")
        elif side == "right":
            c.create_rectangle(cw * (1 - depth), 0, cw, ch, fill=col, outline="")

    def _s_mountain_range(self, c, cw, base_y, count, s):
        spacing = cw / (count + 1)
        for i in range(count):
            mx = spacing * (i + 1) + ((i * 37) % 20 - 10) * s
            mh = (60 + (i * 23) % 40) * s
            mw = (40 + (i * 17) % 30) * s
            c.create_polygon(mx - mw * 1.3, base_y, mx, base_y - mh,
                             mx + mw * 1.3, base_y, fill="#0a140a", outline="")
            c.create_polygon(mx - mw, base_y, mx + mw * 0.1,
                             base_y - mh * 0.8, mx + mw, base_y,
                             fill="#0e1a0e", outline="")
            c.create_polygon(mx - mw * 0.15, base_y - mh * 0.65,
                             mx + mw * 0.1, base_y - mh * 0.8,
                             mx + mw * 0.25, base_y - mh * 0.65,
                             fill="#c0c0c8", outline="")

    def _s_building_row(self, c, cw, ch, side, count, s):
        if side in ("top", "bottom"):
            is_top = side == "top"
            spacing = cw / (count + 1)
            for i in range(count):
                bw = (14 + (i * 13) % 12) * s
                bh = (25 + (i * 31) % 45) * s
                bx = spacing * (i + 1) - bw / 2
                y1 = 0 if is_top else ch - bh
                y2 = bh if is_top else ch
                c.create_rectangle(bx, y1, bx + bw, y2,
                                   fill="#0e0e18", outline="#1a1a28")
                for wy in range(max(1, int(bh / (7 * s)))):
                    for wx in range(max(1, int(bw / (5 * s)))):
                        if (wx + wy + i) % 3 != 0:
                            c.create_rectangle(
                                bx + wx * 5 * s + 2 * s,
                                y1 + wy * 7 * s + 2 * s,
                                bx + wx * 5 * s + 4 * s,
                                y1 + wy * 7 * s + 5 * s,
                                fill="#252540", outline="")
        elif side in ("left", "right"):
            is_left = side == "left"
            spacing = ch / (count + 1)
            for i in range(count):
                bw = (20 + (i * 31) % 35) * s
                bh = (10 + (i * 13) % 8) * s
                by = spacing * (i + 1) - bh / 2
                x1 = 0 if is_left else cw - bw
                x2 = bw if is_left else cw
                c.create_rectangle(x1, by, x2, by + bh,
                                   fill="#0e0e18", outline="#1a1a28")

    def _s_dune_field(self, c, cw, ch, count, s):
        for i in range(count):
            dx = cw * (0.1 + 0.8 * i / max(count, 1))
            dx += ((i * 47) % 30 - 15) * s
            dy = ch * (0.85 + (i * 13) % 5 * 0.02)
            dw = (60 + (i * 31) % 40) * s
            dh = (15 + (i * 17) % 10) * s
            c.create_arc(dx - dw, dy - dh, dx + dw, dy + dh,
                         start=0, extent=180, fill="#1a1608",
                         outline="#201c0a", style=tk.PIESLICE)

    def _s_vegas_strip(self, c, cw, ch, s):
        neon = ["#ff1060", "#4040ff", "#ff8020",
                "#20ff60", "#ff20ff", "#20ffff"]
        strip_y = ch * 0.04
        for i in range(8):
            bw = (15 + (i * 7) % 12) * s
            bh = (30 + (i * 23) % 50) * s
            bx = cw * 0.08 + i * cw * 0.11
            c.create_rectangle(bx, strip_y, bx + bw, strip_y + bh,
                               fill="#0e0e1a", outline="#1a1a30")
            nc = neon[i % len(neon)]
            c.create_line(bx, strip_y, bx + bw, strip_y,
                          fill=nc, width=max(1, 2 * s))
            for wy in range(max(1, int(bh / (8 * s)))):
                for wx in range(max(1, int(bw / (5 * s)))):
                    if (wx + wy + i) % 2 == 0:
                        wc = neon[(wx + wy + i) % len(neon)]
                        c.create_rectangle(
                            bx + wx * 5 * s + 1 * s,
                            strip_y + wy * 8 * s + 2 * s,
                            bx + wx * 5 * s + 3 * s,
                            strip_y + wy * 8 * s + 5 * s,
                            fill=wc, outline="")

    def _s_sphere(self, c, x, y, r):
        for i in range(4, 0, -1):
            gr = r + i * 5
            c.create_oval(x - gr, y - gr, x + gr, y + gr,
                          fill="", outline="#2020a0", width=1)
        c.create_oval(x - r, y - r, x + r, y + r,
                      fill="#0a0a30", outline="#4040c0", width=2)
        for f in [-0.6, -0.3, 0, 0.3, 0.6]:
            dy = r * f
            hw_v = math.sqrt(max(0, r * r - dy * dy))
            c.create_line(x - hw_v, y + dy, x + hw_v, y + dy,
                          fill="#3030a0", width=1)
        for f in [-0.3, 0, 0.3]:
            dx = r * f
            hh = math.sqrt(max(0, r * r - dx * dx))
            c.create_line(x + dx, y - hh, x + dx, y + hh,
                          fill="#3030a0", width=1)
        c.create_text(x, y + r + 8, text="SPHERE",
                      font=("Helvetica Neue", max(6, int(r * 0.3)), "bold"),
                      fill="#4040c0")

    def _s_ferris(self, c, x, y, r):
        c.create_line(x, y + r + 10, x - r * 0.3, y,
                      fill="#2a2a34", width=2)
        c.create_line(x, y + r + 10, x + r * 0.3, y,
                      fill="#2a2a34", width=2)
        c.create_oval(x - r, y - r, x + r, y + r,
                      fill="", outline="#3a3a4a", width=2)
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            c.create_line(x, y, x + math.cos(rad) * r,
                          y + math.sin(rad) * r,
                          fill="#2a2a3a", width=1)
            gx = x + math.cos(rad) * r
            gy = y + math.sin(rad) * r
            gr = max(2, r * 0.08)
            c.create_oval(gx - gr, gy - gr, gx + gr, gy + gr,
                          fill="#3a3a5a", outline="#4a4a6a")
        hr = max(3, r * 0.1)
        c.create_oval(x - hr, y - hr, x + hr, y + hr,
                      fill="#3a3a4a", outline="")

    def _s_yacht_harbor(self, c, cw, ch, count, s):
        water_y = ch * 0.88
        for i in range(count):
            yx = cw * (0.2 + 0.6 * i / max(count, 1))
            yy = water_y + 10 * s + (i * 7) % 3 * 5 * s
            c.create_polygon(yx - 12 * s, yy, yx - 15 * s, yy - 4 * s,
                             yx + 15 * s, yy - 4 * s, yx + 12 * s, yy,
                             fill="#1a1a28", outline="#2a2a3a")
            c.create_line(yx, yy - 4 * s, yx, yy - 20 * s,
                          fill="#2a2a3a", width=1)
            c.create_polygon(yx, yy - 18 * s, yx + 8 * s, yy - 6 * s,
                             yx, yy - 6 * s,
                             fill="#16162a", outline="#2a2a3a")

    def _s_lake(self, c, cx, cy, rw, rh):
        c.create_oval(cx - rw, cy - rh, cx + rw, cy + rh,
                      fill="#0a1428", outline="#122035", width=1)
        for f in [0.4, 0.7]:
            c.create_oval(cx - rw * f, cy - rh * f,
                          cx + rw * f, cy + rh * f,
                          fill="", outline="#122a3a", width=1, dash=(10, 15))

    def _s_tower(self, c, x, y, s):
        c.create_polygon(x - 8 * s, y + 50 * s, x - 3 * s, y,
                         x + 3 * s, y, x + 8 * s, y + 50 * s,
                         fill="#14141e", outline="#1e1e2a")
        c.create_rectangle(x - 12 * s, y - 3 * s, x + 12 * s, y + 3 * s,
                           fill="#1a1a28", outline="#2a2a3a")
        c.create_line(x, y - 3 * s, x, y - 12 * s,
                      fill="#2a2a3a", width=max(1, 2 * s))

    def _s_stadium(self, c, track, hw, frac, side, s):
        num = len(track)
        base = int(frac * num) % num
        span = max(15, int(num * 0.10))
        for row in range(6):
            inner_d = hw * 2.4 + row * 7 * s
            outer_d = inner_d + 6 * s
            pts = []
            for j in range(-span // 2, span // 2 + 1, 2):
                idx = (base + j) % num
                nx, ny = self._track_normal(track, idx)
                pts.extend([track[idx][0] + nx * inner_d * side,
                            track[idx][1] + ny * inner_d * side])
            for j in range(span // 2, -span // 2 - 1, -2):
                idx = (base + j) % num
                nx, ny = self._track_normal(track, idx)
                pts.extend([track[idx][0] + nx * outer_d * side,
                            track[idx][1] + ny * outer_d * side])
            if len(pts) >= 6:
                gray = 0x10 + row * 3
                c.create_polygon(pts,
                                 fill=f"#{gray:02x}{gray:02x}{gray + 6:02x}",
                                 outline="#1e1e28", smooth=True)

    def _s_cactus_scatter(self, c, track, hw, s):
        num = len(track)
        g = "#1a3a1a"
        for i in range(0, num, 35):
            nx, ny = self._track_normal(track, i)
            for side in (1, -1):
                if (i + (1 if side > 0 else 0)) % 3 == 0:
                    continue
                dist = hw * 3.0 + ((i * 13) % 20) * s
                cx = track[i][0] + nx * dist * side
                cy = track[i][1] + ny * dist * side
                cs = s * (0.5 + ((i * 7) % 5) * 0.1)
                c.create_line(cx, cy, cx, cy - 15 * cs,
                              fill=g, width=max(2, 3 * cs))
                c.create_line(cx - 6 * cs, cy - 12 * cs,
                              cx - 6 * cs, cy - 8 * cs,
                              fill=g, width=max(1, 2 * cs))
                c.create_line(cx - 6 * cs, cy - 8 * cs, cx, cy - 8 * cs,
                              fill=g, width=max(1, 2 * cs))
                c.create_line(cx + 5 * cs, cy - 10 * cs,
                              cx + 5 * cs, cy - 6 * cs,
                              fill=g, width=max(1, 2 * cs))
                c.create_line(cx + 5 * cs, cy - 6 * cs, cx, cy - 6 * cs,
                              fill=g, width=max(1, 2 * cs))

    # ── Draw real track ──

    def _draw_real_track(self, cw, ch, raw_pts, preds):
        canvas = self.track_canvas
        margin = 80

        scaled = [(margin + x * (cw - 2 * margin),
                    margin + y * (ch - 2 * margin)) for x, y in raw_pts]

        track = self._interpolate_track(scaled, 500)
        self._track_pts = track
        num = len(track)
        tw = max(28, min(48, int(min(cw, ch) * 0.045)))
        hw = tw / 2

        # Carbon fiber background — tiled PIL image, built bottom-up for speed
        if HAS_PIL:
            tile = _make_carbon_fiber_img(6, 6)
            if tile:
                row = Image.new("RGB", (cw, 6), (10, 10, 11))
                for tx in range(0, cw, 6):
                    row.paste(tile, (tx, 0))
                bg_img = Image.new("RGB", (cw, ch), (10, 10, 11))
                for ty in range(0, ch, 6):
                    bg_img.paste(row, (0, ty))
                self._cf_bg = ImageTk.PhotoImage(bg_img)
                canvas.create_image(0, 0, image=self._cf_bg, anchor="nw")
                self._tk_images.append(self._cf_bg)

        self._draw_scene(canvas, cw, ch, track, hw, self._viz_circuit)

        # ── Track surface ──
        flat = []
        for p in track:
            flat.extend(p)
        canvas.create_polygon(flat, outline="#16161e", fill="", width=tw + 4, smooth=True)
        canvas.create_polygon(flat, outline="#111119", fill="", width=tw, smooth=True)

        # ── Borders ──
        for side_sign in (1, -1):
            border = []
            for i in range(num):
                nx, ny = self._track_normal(track, i)
                border.extend([track[i][0] + nx * hw * side_sign,
                               track[i][1] + ny * hw * side_sign])
            canvas.create_line(border, fill="#2a2a38", width=1.5, smooth=True)

        # ── Curvature analysis ──
        curvatures = []
        step = max(3, num // 150)
        for i in range(num):
            p_prev = track[(i - step) % num]
            p_cur = track[i]
            p_next = track[(i + step) % num]
            dx1, dy1 = p_cur[0] - p_prev[0], p_cur[1] - p_prev[1]
            dx2, dy2 = p_next[0] - p_cur[0], p_next[1] - p_cur[1]
            curvatures.append(abs(dx1 * dy2 - dy1 * dx2))

        curv_sorted = sorted(curvatures, reverse=True)
        curv_top = curv_sorted[min(30, num - 1)]

        # ── Kerbs at tight corners ──
        for i, c in enumerate(curvatures):
            if c >= curv_top and i % 6 == 0:
                nx, ny = self._track_normal(track, i)
                for ss in (1, -1):
                    kx = track[i][0] + nx * hw * ss
                    ky = track[i][1] + ny * hw * ss
                    canvas.create_oval(kx - 2, ky - 2, kx + 2, ky + 2,
                                       fill=RED, outline="")

        # ── Start / finish ──
        sf = track[0]
        nx, ny = self._track_normal(track, 0)
        canvas.create_line(sf[0] + nx * hw, sf[1] + ny * hw,
                           sf[0] - nx * hw, sf[1] - ny * hw,
                           fill=WHITE, width=3, dash=(4, 4))
        canvas.create_text(sf[0] + nx * (hw + 16), sf[1] + ny * (hw + 16),
                           text="START / FINISH", font=("Helvetica Neue", 7, "bold"),
                           fill=MUTED)

        # ── MOM zones — find the two longest straight stretches ──
        low_curv_threshold = curv_sorted[num // 2] if num > 10 else 0
        runs = []
        run_start = None
        for i in range(num * 2):
            idx = i % num
            if curvatures[idx] < low_curv_threshold:
                if run_start is None:
                    run_start = idx
            else:
                if run_start is not None:
                    run_len = (idx - run_start) % num
                    runs.append((run_start, run_len))
                    run_start = None
            if i >= num and run_start is not None:
                run_len = (idx - run_start) % num
                runs.append((run_start, run_len))
                break

        runs_clean = [(s, l) for s, l in runs if l >= 12]
        runs_clean.sort(key=lambda x: -x[1])

        mom_placed = []
        for s, l in runs_clean:
            mid = (s + l // 2) % num
            if all(min(abs(mid - prev), num - abs(mid - prev)) > num // 5 for prev in mom_placed):
                mom_placed.append(mid)
                zone_flat = []
                for j in range(l):
                    zone_flat.extend(track[(s + j) % num])
                if len(zone_flat) >= 4:
                    canvas.create_line(zone_flat, fill=GREEN, width=3, smooth=True, dash=(8, 4))
                mnx, mny = self._normal_at(float(mid))
                mp = self._pos_at(float(mid))
                canvas.create_text(mp[0] + mnx * (hw + 16), mp[1] + mny * (hw + 16),
                                   text="MOM", font=("Helvetica Neue", 8, "bold"), fill=GREEN)
            if len(mom_placed) >= 2:
                break

        # ── Center podium ──
        xs = [p[0] for p in track]
        ys = [p[1] for p in track]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        canvas.create_text(cx, cy - 48, text=self._viz_circuit,
                           font=("Helvetica Neue", 14, "bold"), fill=MUTED)
        canvas.create_text(cx, cy - 30, text="━━  PREDICTED PODIUM  ━━",
                           font=("Helvetica Neue", 8), fill=MUTED)

        p1 = preds[0]
        canvas.create_text(cx, cy - 12, text="🏆",
                           font=("Helvetica Neue", 18))
        canvas.create_text(cx, cy + 12,
                           text=f"P1  {p1['abbreviation']}",
                           font=("Helvetica Neue", 20, "bold"),
                           fill=tc(p1["team"]))
        canvas.create_text(cx, cy + 32,
                           text=f"{p1['probability']*100:.1f}%  ·  {p1['team']}",
                           font=("Helvetica Neue", 10), fill=RED)

        if len(preds) >= 3:
            p2, p3 = preds[1], preds[2]
            gap = 85
            canvas.create_text(cx - gap, cy + 54,
                               text=f"P2  {p2['abbreviation']}",
                               font=("Helvetica Neue", 13, "bold"),
                               fill=tc(p2["team"]))
            canvas.create_text(cx - gap, cy + 70,
                               text=f"{p2['probability']*100:.1f}%",
                               font=("Helvetica Neue", 9), fill=MUTED)
            canvas.create_text(cx + gap, cy + 54,
                               text=f"P3  {p3['abbreviation']}",
                               font=("Helvetica Neue", 13, "bold"),
                               fill=tc(p3["team"]))
            canvas.create_text(cx + gap, cy + 70,
                               text=f"{p3['probability']*100:.1f}%",
                               font=("Helvetica Neue", 9), fill=MUTED)

        # ── Animation — only top 8 drivers for performance ──
        max_anim = min(8, len(preds))
        n = max_anim
        lead_gap = num * 0.65
        spacing = lead_gap / max(n, 1)
        self._anim_targets = [lead_gap - i * spacing for i in range(n)]
        self._anim_pos = [0.0] * n
        self._anim_frame = 0
        self._anim_running = True
        self._anim_hw = hw
        self._anim_num = float(num)
        self._anim_count = n

        self._viz_logos = []
        for i, p in enumerate(preds[:max_anim]):
            if HAS_PIL:
                logo_img = load_logo(p["team"], 18)
                if logo_img:
                    tkimg = ImageTk.PhotoImage(logo_img)
                    self._tk_images.append(tkimg)
                    self._viz_logos.append(tkimg)
                else:
                    self._viz_logos.append(None)
            else:
                self._viz_logos.append(None)

        sf = track[0]
        self._dot_ids = []
        self._dot_txt_ids = []
        self._trail_ids = []
        self._label_ids = []
        self._label_visible = [False] * n

        self._glow_id = canvas.create_oval(0, 0, 0, 0, fill="", outline=GOLD_GLOW,
                                            width=2, state="hidden")

        for i in range(max_anim):
            p = preds[i]
            color = tc(p["team"])
            dot_r = 8 if i == 0 else 6 if i < 3 else 4

            dot = canvas.create_oval(
                sf[0] - dot_r, sf[1] - dot_r, sf[0] + dot_r, sf[1] + dot_r,
                fill=color, outline=WHITE if i == 0 else color,
                width=2 if i == 0 else 1)
            self._dot_ids.append(dot)

            txt = canvas.create_text(sf[0], sf[1], text=str(i + 1),
                                     font=("Helvetica Neue", 7, "bold"),
                                     fill=WHITE if i < 3 else BG)
            self._dot_txt_ids.append(txt)

            # Trails only for top 3
            trails = []
            if i < 3:
                for t_off in range(1, 3):
                    tr_r = dot_r * (1 - t_off * 0.3)
                    tr = canvas.create_oval(0, 0, 0, 0, fill=color, outline="",
                                            stipple="gray25" if t_off == 1 else "gray12",
                                            state="hidden")
                    trails.append(tr_r)
                    trails.append(tr)
            self._trail_ids.append(trails)

            bg_fill = GOLD_DIM if i == 0 else BG_CARD
            border_w = 2 if i < 3 else 1
            name_color = WHITE if i < 3 else GRAY
            prob_color = GOLD_GLOW if i == 0 else MUTED

            connector = canvas.create_line(0, 0, 0, 0, fill=color, width=1,
                                           dash=(2, 2), state="hidden")
            card = canvas.create_rectangle(0, 0, 0, 0, fill=bg_fill, outline=color,
                                           width=border_w, state="hidden")
            logo_item = None
            if i < len(self._viz_logos) and self._viz_logos[i]:
                logo_item = canvas.create_image(0, 0, image=self._viz_logos[i],
                                                anchor="center", state="hidden")
            name_txt = canvas.create_text(0, 0, text=p["abbreviation"],
                                          font=("Helvetica Neue", 10 if i == 0 else 9, "bold"),
                                          fill=name_color, anchor="w", state="hidden")
            prob_txt = canvas.create_text(0, 0,
                                          text=f"P{i+1}  {p['probability']*100:.1f}%",
                                          font=("Helvetica Neue", 8),
                                          fill=prob_color, anchor="w", state="hidden")
            self._label_ids.append({
                "connector": connector, "card": card, "logo": logo_item,
                "name": name_txt, "prob": prob_txt
            })

        self._create_scene_anims(canvas, cw, ch)
        self._anim_tick()

    # ── Circuit-specific animated decorations ──

    SCENE_ANIMS = {
        "Albert Park": [("kangaroo", 2)],
        "Suzuka": [("blossom", 15)],
        "Shanghai": [("blossom", 8)],
        "Miami Autodrome": [("sparkle_water", 6)],
        "Monaco": [("sparkle_water", 8)],
        "Las Vegas Strip": [("firework", 3), ("neon_flash", 5)],
        "Sakhir": [("star", 10)],
        "Lusail": [("star", 10)],
        "Spa-Francorchamps": [("rain", 15)],
        "Zandvoort": [("seagull", 2)],
        "Marina Bay": [("firework", 2), ("sparkle_water", 5)],
        "Yas Marina": [("sparkle_water", 5), ("star", 6)],
        "Interlagos": [("blossom", 6)],
        "Jeddah Corniche": [("star", 8)],
        "Baku City Circuit": [("seagull", 2)],
        "Silverstone": [("rain", 8)],
        "Red Bull Ring": [("seagull", 2)],
        "COTA": [("star", 5)],
    }

    def _create_scene_anims(self, canvas, cw, ch):
        import random as _rng
        self._rng = _rng
        self._scene_items = []
        anims = self.SCENE_ANIMS.get(self._viz_circuit, [])

        for atype, count in anims:
            for _ in range(count):
                x = _rng.uniform(40, cw - 40)
                y = _rng.uniform(40, ch - 40)

                if atype == "blossom":
                    size = _rng.uniform(3, 6)
                    petal = canvas.create_oval(x, y, x + size, y + size,
                                               fill="#ffb7c5", outline="#ff8fa3", width=1)
                    self._scene_items.append({
                        "type": "blossom", "id": petal,
                        "x": x, "y": y, "size": size,
                        "vx": _rng.uniform(-0.3, 0.3),
                        "vy": _rng.uniform(0.4, 1.0),
                        "sway": _rng.uniform(0, 6.28),
                        "cw": cw, "ch": ch,
                    })

                elif atype == "kangaroo":
                    x = _rng.uniform(60, cw - 60)
                    y = _rng.uniform(ch * 0.3, ch * 0.7)
                    body = canvas.create_oval(x - 8, y - 5, x + 8, y + 5,
                                               fill="#8B6914", outline="#6B4F12")
                    head = canvas.create_oval(x + 5, y - 10, x + 13, y - 2,
                                               fill="#8B6914", outline="#6B4F12")
                    ear1 = canvas.create_oval(x + 7, y - 14, x + 10, y - 9,
                                               fill="#A07818", outline="#6B4F12")
                    ear2 = canvas.create_oval(x + 10, y - 14, x + 13, y - 9,
                                               fill="#A07818", outline="#6B4F12")
                    tail = canvas.create_line(x - 8, y, x - 18, y - 6,
                                               fill="#6B4F12", width=2)
                    self._scene_items.append({
                        "type": "kangaroo",
                        "ids": [body, head, ear1, ear2, tail],
                        "base_x": x, "base_y": y,
                        "phase": _rng.uniform(0, 6.28),
                        "speed": _rng.uniform(0.03, 0.06),
                        "hop_h": _rng.uniform(8, 16),
                        "direction": _rng.choice([-1, 1]),
                    })

                elif atype == "firework":
                    sparks = []
                    cx = _rng.uniform(cw * 0.2, cw * 0.8)
                    cy = _rng.uniform(ch * 0.1, ch * 0.4)
                    color = _rng.choice(["#ff4444", "#44ff44", "#ffaa00",
                                          "#ff66ff", "#44aaff", GOLD_GLOW])
                    for j in range(8):
                        angle = j * 0.785
                        sid = canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2,
                                                  fill=color, outline="", state="hidden")
                        sparks.append({"id": sid, "angle": angle})
                    self._scene_items.append({
                        "type": "firework", "sparks": sparks,
                        "cx": cx, "cy": cy, "color": color,
                        "phase": _rng.uniform(0, 200),
                        "period": _rng.uniform(120, 250),
                        "cw": cw, "ch": ch,
                    })

                elif atype == "star":
                    x = _rng.uniform(20, cw - 20)
                    y = _rng.uniform(10, ch * 0.35)
                    star = canvas.create_oval(x - 1, y - 1, x + 1, y + 1,
                                               fill="#ffffff", outline="")
                    self._scene_items.append({
                        "type": "star", "id": star,
                        "x": x, "y": y,
                        "phase": _rng.uniform(0, 6.28),
                        "speed": _rng.uniform(0.02, 0.08),
                        "base_size": _rng.uniform(0.5, 2.0),
                    })

                elif atype == "rain":
                    x = _rng.uniform(0, cw)
                    y = _rng.uniform(-20, ch)
                    drop = canvas.create_line(x, y, x - 1, y + 8,
                                               fill="#4488aa", width=1)
                    self._scene_items.append({
                        "type": "rain", "id": drop,
                        "x": x, "y": y,
                        "speed": _rng.uniform(3, 6),
                        "cw": cw, "ch": ch,
                    })

                elif atype == "sparkle_water":
                    x = _rng.uniform(20, cw - 20)
                    y = _rng.uniform(ch * 0.6, ch - 30)
                    sp = canvas.create_oval(x - 1, y - 1, x + 1, y + 1,
                                             fill="#aaddff", outline="")
                    self._scene_items.append({
                        "type": "sparkle_water", "id": sp,
                        "x": x, "y": y,
                        "phase": _rng.uniform(0, 6.28),
                        "speed": _rng.uniform(0.04, 0.10),
                    })

                elif atype == "seagull":
                    x = _rng.uniform(20, cw - 80)
                    y = _rng.uniform(20, ch * 0.3)
                    wing1 = canvas.create_line(x, y, x - 8, y - 4, fill="white", width=1)
                    wing2 = canvas.create_line(x, y, x + 8, y - 4, fill="white", width=1)
                    self._scene_items.append({
                        "type": "seagull",
                        "ids": [wing1, wing2],
                        "x": x, "y": y,
                        "vx": _rng.uniform(0.5, 1.2),
                        "phase": _rng.uniform(0, 6.28),
                        "cw": cw, "ch": ch,
                    })

                elif atype == "neon_flash":
                    x = _rng.uniform(cw * 0.15, cw * 0.85)
                    y = _rng.uniform(ch * 0.15, ch * 0.55)
                    color = _rng.choice(["#ff0066", "#00ffcc", "#ff6600",
                                          "#cc00ff", "#00ff66", "#ffcc00"])
                    neon = canvas.create_rectangle(x, y, x + _rng.uniform(12, 25),
                                                    y + _rng.uniform(3, 6),
                                                    fill=color, outline="", state="hidden")
                    self._scene_items.append({
                        "type": "neon_flash", "id": neon,
                        "phase": _rng.uniform(0, 200),
                        "on_time": _rng.uniform(15, 40),
                        "off_time": _rng.uniform(30, 80),
                        "color": color,
                    })

    def _tick_scene_anims(self, frame):
        canvas = self.track_canvas
        for item in self._scene_items:
            t = item["type"]

            if t == "blossom":
                item["x"] += item["vx"] + math.sin(item["sway"] + frame * 0.03) * 0.4
                item["y"] += item["vy"]
                if item["y"] > item["ch"] + 10:
                    item["y"] = -10
                    item["x"] = self._rng.uniform(20, item["cw"] - 20)
                s = item["size"]
                wobble = math.sin(item["sway"] + frame * 0.05) * 2
                canvas.coords(item["id"], item["x"] + wobble, item["y"],
                              item["x"] + s + wobble, item["y"] + s)

            elif t == "kangaroo":
                item["phase"] += item["speed"]
                hop = abs(math.sin(item["phase"])) * item["hop_h"]
                drift = math.sin(item["phase"] * 0.3) * 25
                dx = drift * item["direction"]
                bx, by = item["base_x"] + dx, item["base_y"] - hop
                offsets = [(bx - 8, by - 5, bx + 8, by + 5),
                           (bx + 5, by - 10, bx + 13, by - 2),
                           (bx + 7, by - 14, bx + 10, by - 9),
                           (bx + 10, by - 14, bx + 13, by - 9)]
                for idx, coords in enumerate(offsets):
                    canvas.coords(item["ids"][idx], *coords)
                canvas.coords(item["ids"][4], bx - 8, by, bx - 18, by - 6 + hop * 0.3)

            elif t == "firework":
                cycle = (frame + item["phase"]) % item["period"]
                burst_dur = 40
                if cycle < burst_dur:
                    t_frac = cycle / burst_dur
                    radius = t_frac * 30
                    alpha_state = "normal" if t_frac < 0.8 else "hidden"
                    for spark in item["sparks"]:
                        sx = item["cx"] + math.cos(spark["angle"]) * radius
                        sy = item["cy"] + math.sin(spark["angle"]) * radius
                        sr = 3 * (1 - t_frac * 0.5)
                        canvas.coords(spark["id"], sx - sr, sy - sr, sx + sr, sy + sr)
                        canvas.itemconfigure(spark["id"], state=alpha_state)
                else:
                    for spark in item["sparks"]:
                        canvas.itemconfigure(spark["id"], state="hidden")

            elif t == "star":
                brightness = 0.3 + 0.7 * abs(math.sin(item["phase"] + frame * item["speed"]))
                r = item["base_size"] * (0.5 + brightness * 0.5)
                gray = int(150 + brightness * 105)
                color = f"#{gray:02x}{gray:02x}{min(255, gray + 20):02x}"
                canvas.coords(item["id"], item["x"] - r, item["y"] - r,
                              item["x"] + r, item["y"] + r)
                canvas.itemconfigure(item["id"], fill=color)

            elif t == "rain":
                item["y"] += item["speed"]
                item["x"] -= item["speed"] * 0.3
                if item["y"] > item["ch"]:
                    item["y"] = -10
                    item["x"] = self._rng.uniform(0, item["cw"])
                canvas.coords(item["id"], item["x"], item["y"],
                              item["x"] - 1, item["y"] + 8)

            elif t == "sparkle_water":
                brightness = abs(math.sin(item["phase"] + frame * item["speed"]))
                r = 1 + brightness * 2
                gray = int(100 + brightness * 155)
                color = f"#{max(80,gray-30):02x}{gray:02x}{min(255,gray+30):02x}"
                canvas.coords(item["id"], item["x"] - r, item["y"] - r,
                              item["x"] + r, item["y"] + r)
                canvas.itemconfigure(item["id"], fill=color)

            elif t == "seagull":
                item["x"] += item["vx"]
                item["phase"] += 0.08
                wing_y = math.sin(item["phase"]) * 5
                if item["x"] > item["cw"] + 20:
                    item["x"] = -20
                x, y = item["x"], item["y"]
                canvas.coords(item["ids"][0], x, y, x - 8, y - 4 + wing_y)
                canvas.coords(item["ids"][1], x, y, x + 8, y - 4 + wing_y)

            elif t == "neon_flash":
                cycle = (frame + item["phase"]) % (item["on_time"] + item["off_time"])
                if cycle < item["on_time"]:
                    canvas.itemconfigure(item["id"], state="normal")
                else:
                    canvas.itemconfigure(item["id"], state="hidden")

    # ── Smooth animation loop ──

    def _anim_tick(self):
        if not self._anim_running or self._current_view != "viz":
            return

        canvas = self.track_canvas
        frame = self._anim_frame
        hw = self._anim_hw
        num = self._anim_num
        n = self._anim_count

        spread_duration = 90.0
        orbit_speed = 0.35

        for i in range(n):
            target = self._anim_targets[i]

            if frame < spread_duration:
                t = frame / spread_duration
                ease = 1.0 - (1.0 - t) ** 3
                pos = target * ease
            else:
                pos = target + (frame - spread_duration) * orbit_speed

            pos = pos % num
            self._anim_pos[i] = pos
            px, py = self._pos_at(pos)
            dot_r = 8 if i == 0 else 6 if i < 3 else 4

            canvas.coords(self._dot_ids[i],
                          px - dot_r, py - dot_r, px + dot_r, py + dot_r)
            canvas.coords(self._dot_txt_ids[i], px, py)

            trails = self._trail_ids[i]
            if trails and frame >= 8:
                for t_idx in range(0, len(trails), 2):
                    tr_r = trails[t_idx]
                    tr_id = trails[t_idx + 1]
                    trail_pos = (pos - (t_idx // 2 + 1) * 2.5) % num
                    tx, ty = self._pos_at(trail_pos)
                    canvas.coords(tr_id, tx - tr_r, ty - tr_r, tx + tr_r, ty + tr_r)
                    canvas.itemconfigure(tr_id, state="normal")

            if i == 0 and frame >= spread_duration:
                pulse = 0.5 + 0.5 * math.sin(frame * 0.06)
                glow_r = 14 + pulse * 6
                canvas.coords(self._glow_id,
                              px - glow_r, py - glow_r, px + glow_r, py + glow_r)
                canvas.itemconfigure(self._glow_id, state="normal")

            reveal_at = spread_duration + i * 2
            if frame >= reveal_at:
                if not self._label_visible[i]:
                    self._label_visible[i] = True
                    for item in self._label_ids[i].values():
                        if item is not None:
                            canvas.itemconfigure(item, state="normal")

                nx, ny = self._normal_at(pos)
                side = 1.0 if i % 2 == 0 else -1.0
                dist = hw + 30 + (10 if i < 3 else 0)
                lx = px + nx * dist * side
                ly = py + ny * dist * side
                cw2, ch2 = 44.0, 19.0

                ids = self._label_ids[i]
                canvas.coords(ids["connector"], px, py, lx, ly)
                canvas.coords(ids["card"], lx - cw2, ly - ch2, lx + cw2, ly + ch2)
                if ids["logo"] is not None:
                    canvas.coords(ids["logo"], lx - cw2 + 14, ly)
                canvas.coords(ids["name"], lx + 2, ly - 7)
                canvas.coords(ids["prob"], lx + 2, ly + 8)

        # Scene animations: update every 2nd frame for performance
        if frame % 2 == 0 and hasattr(self, "_scene_items") and self._scene_items:
            self._tick_scene_anims(frame)

        self._anim_frame += 1
        self.root.after(40, self._anim_tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ApexAI().run()
