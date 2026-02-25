#!/usr/bin/env python3
"""
ApexAI – F1 winner prediction
"""
import io
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

from prediction import run_predictions, run_predictions_all_races
from team_colors import TEAM_COLORS
from team_logos import load_logo

# -- Theme --
BG = "#06060a"
BG_SURFACE = "#0d0d14"
BG_CARD = "#111118"
BG_HOVER = "#18181f"
BORDER = "#1e1e28"
GOLD = "#d4a926"
GOLD_DIM = "#8a7020"
GOLD_GLOW = "#f5d547"
WHITE = "#f0f0f2"
GRAY = "#8888a0"
MUTED = "#505068"
RED = "#e0443a"
GREEN = "#3ae08a"


def tc(team: str) -> str:
    return TEAM_COLORS.get(team, "#555566")


ALGO_TEXT = """\
# ApexAI · Random Forest Pipeline

features = [
  "Abbreviation",       # VER, HAM, NOR...
  "TeamName",           # Ferrari, McLaren...
  "GridPosition",       # Starting grid slot
  "DriverNumber",       # Car number
  "DriverPointsBefore", # Cumulative driver pts
  "TeamPointsBefore",   # Cumulative team pts
]

preprocess = ColumnTransformer([
  OneHotEncoder → Abbreviation, TeamName
  StandardScaler → numeric features
])

model = RandomForestClassifier(
  n_estimators = 150–250,
  max_depth    = 10–20,
)

tuning = RandomizedSearchCV(
  n_iter=4, cv=3,
  scoring="accuracy",
)

output = model.predict_proba(X)[:, 1]
# → win probability per driver\
"""


class ApexAI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ApexAI")
        self.root.configure(bg=BG)
        self.root.minsize(960, 780)
        self.root.geometry("1120x880")
        self.result = None
        self._logos = {}
        self._build()

    # -- Logo cache --
    def _logo(self, team: str, sz: int = 24):
        k = f"{team}_{sz}"
        if k not in self._logos:
            if HAS_PIL:
                img = load_logo(team, sz)
                self._logos[k] = ImageTk.PhotoImage(img) if img else None
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

        tk.Frame(self.root, bg=GOLD_DIM, height=1).pack(fill=tk.X, padx=36)

        # Controls
        ctrl = tk.Frame(self.root, bg=BG, padx=36, pady=16)
        ctrl.pack(fill=tk.X)

        self.btn_predict = self._make_btn(ctrl, "Predict Next Race", GOLD, BG, self._on_predict)
        self.btn_predict.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_all = self._make_btn(ctrl, "Backtest All Races", BG_CARD, GRAY, self._on_all_races, border=BORDER)
        self.btn_all.pack(side=tk.LEFT, padx=(0, 16))

        self.status_lbl = tk.Label(ctrl, text="", font=("Helvetica Neue", 11), fg=MUTED, bg=BG)
        self.status_lbl.pack(side=tk.LEFT, padx=(8, 0))

        # Body – two panels
        body = tk.Frame(self.root, bg=BG, padx=36, pady=8)
        body.pack(fill=tk.BOTH, expand=True)

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

        # Right panel – algorithm
        right = tk.Frame(body, bg=BG_CARD, width=320, padx=20, pady=20)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        right.configure(highlightbackground=BORDER, highlightthickness=1)

        tk.Label(right, text="ALGORITHM", font=("Helvetica Neue", 10, "bold"), fg=GOLD, bg=BG_CARD).pack(anchor="w")
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
        b = tk.Button(parent, text=text, font=("Helvetica Neue", 11, "bold"), fg=fg_c, bg=bg_c, activeforeground=fg_c, activebackground=bg_c, relief=tk.FLAT, padx=20, pady=10, cursor="hand2", command=cmd)
        if border:
            b.configure(highlightbackground=border, highlightthickness=1)
        b.bind("<Enter>", lambda e, btn=b, c=bg_c: btn.configure(bg=self._lighten(c)))
        b.bind("<Leave>", lambda e, btn=b, c=bg_c: btn.configure(bg=c))
        return b

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
        st = tk.DISABLED if busy else tk.NORMAL
        self.btn_predict.configure(state=st)
        self.btn_all.configure(state=st)

    # -- Predict next race --
    def _on_predict(self):
        self._set_busy(True)
        self.btn_predict.configure(text="Running...")
        self._set_status("Loading data...")
        self._clear()
        self._show_empty("Running predictions...")

        def work():
            r = run_predictions(progress_callback=self._set_status)
            self.root.after(0, lambda: self._show_predictions(r))

        threading.Thread(target=work, daemon=True).start()

    def _show_predictions(self, r):
        self._set_busy(False)
        self.btn_predict.configure(text="Predict Next Race")
        if "error" in r:
            self._set_status(f"Error: {r['error'][:80]}")
            self._show_empty(f"Error\n\n{r['error'][:300]}")
            return

        self._set_status(f"Accuracy {r['accuracy']:.1%}")
        self._render_chart(r.get("feature_importance", {}))
        self._clear()

        nr = r["next_race"]
        w = nr["predictions"][0]

        # -- Winner banner --
        banner = tk.Frame(self.results, bg=BG_CARD, padx=24, pady=20)
        banner.configure(highlightbackground=BORDER, highlightthickness=1)
        banner.pack(fill=tk.X, pady=(0, 16))

        tk.Label(banner, text=f"NEXT RACE · Round {nr['round']} ({nr['year']})", font=("Helvetica Neue", 10, "bold"), fg=GOLD, bg=BG_CARD).pack(anchor="w", pady=(0, 12))

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
        tk.Label(lr_card, text=f"LAST RACE · Round {lr['round']} ({lr['year']})", font=("Helvetica Neue", 10, "bold"), fg=GOLD, bg=BG_CARD).pack(anchor="w", pady=(0, 10))

        pred, actual = lr["predicted_winner"], lr["actual_winner"]
        hit = pred == actual
        tk.Label(lr_card, text=f"Predicted: {pred}  ·  Actual: {actual}  {'✓' if hit else '✗'}", font=("Helvetica Neue", 11, "bold"), fg=GREEN if hit else RED, bg=BG_CARD).pack(anchor="w", pady=(0, 10))

        for i, p in enumerate(lr["predictions"][:10]):
            self._driver_row(lr_card, i + 1, p["abbreviation"], p["team"], p["probability"], highlight=(p["abbreviation"] == actual))

        # Accuracy footer
        tk.Label(self.results, text=f"Model accuracy: {r['accuracy']:.1%}", font=("Helvetica Neue", 10), fg=MUTED, bg=BG).pack(anchor="w", pady=(8, 20))

    # -- Backtest all races --
    def _on_all_races(self):
        self._set_busy(True)
        self.btn_all.configure(text="Running...")
        self._set_status("Backtesting every race...")
        self._clear()
        self._show_empty("Running backtest on every race...\nThis takes a few minutes.")

        def work():
            r = run_predictions_all_races(progress_callback=self._set_status)
            self.root.after(0, lambda: self._show_all_races(r))

        threading.Thread(target=work, daemon=True).start()

    def _show_all_races(self, r):
        self._set_busy(False)
        self.btn_all.configure(text="Backtest All Races")
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
        for txt, w in [("Year", 5), ("Rnd", 4), ("Predicted", 10), ("Actual", 10), ("", 3)]:
            tk.Label(hdr, text=txt, font=("Helvetica Neue", 9, "bold"), fg=MUTED, bg=BG_SURFACE, width=w, anchor="w").pack(side=tk.LEFT, padx=2)

        for race in r["all_races"]:
            row = tk.Frame(card, bg=BG_CARD, padx=10, pady=4)
            row.pack(fill=tk.X)
            fg = GREEN if race["correct"] else RED
            mark = "✓" if race["correct"] else "✗"
            for txt, w in [(str(race["year"]), 5), (f"R{race['round']}", 4), (race["predicted"], 10), (race["actual"], 10), (mark, 3)]:
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

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ApexAI().run()
