# =========================================
# SCRIPT NAME: replay.py
# PURPOSE:     Allows user to see a dataset, from get_stockdataset.py or test_policy.py
# AUTHOR:      Keone Leao
# DATE:        04/21/26
# DEPENDENCIES:pandas, pickle, matplotlib.pyplot, matplotlib, matplotlib.dates, numpy, matplotlib.ticker
# =========================================

## Imports
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")   # or "Agg" if no GUI is available
import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import FuncFormatter


# ----------------------------
# LOAD DATA
# ----------------------------
def load_results(path):
    with open(path, "rb") as f:
        df = pickle.load(f)

    df = df.copy()

    # --- ensure correct data types ---
    df["Action"] = df["Action"].astype(int)

    # optional: ensure new features are clean
    for col in ["return_1", "return_3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Forward-fill SR-related values safely
    df["SR_rel"] = df["SR_rel"].ffill()
    df["SR_exists"] = df["SR_exists"].fillna(0)

    # make sure index is datetime
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.set_index("Time")
    else:
        # fallback if no Time column exists
        df.index = pd.date_range("2020-01-01", periods=len(df), freq="min")

    return df


# ----------------------------
# REPLAY CLASS
# ----------------------------
class ReplayEngine:

    def __init__(self, df, delay=200):

        self.sr_lines = []

        self.df = df
        self.delay = delay

        self.i = 0
        self.paused = False

        # GUI state
        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        if isinstance(self.ax, np.ndarray):
            self.ax = self.ax[0]

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.trade_lines = []
        self.trade_markers = []

        self.candle_width = None

    # ------------------------
    # KEY CONTROLS
    # ------------------------
    def on_key(self, event):
        print("KEY:", event.key)

        if event.key == " ": # pause play
            self.paused = not self.paused

        elif event.key == "right": # step forward a candle stick
            self.step()

        elif event.key == "left":
            self.i = max(50, self.i - 1)
            self.step()

        self.fig.canvas.draw_idle()

    # ------------------------
    # INITIAL SETUP
    # ------------------------
    def setup(self):

        self.ax.set_facecolor("#131722")
        self.ax.grid(True, alpha=0.2)
        self.ax.set_title("Trading Replay Engine", color="white")
        self.setup_axes()

        # ---- FIX: initialize candle width FIRST ----
        if len(self.df) > 1:
            dates = mdates.date2num(self.df.index[:2])
            self.candle_width = (dates[1] - dates[0]) * 0.8
        else:
            self.candle_width = 0.01




    

    # ------------------------
    # AXIS STYLE
    # ------------------------
    def setup_axes(self):

        self.ax.set_facecolor("#131722")
        self.fig.patch.set_facecolor("#131722")

        # ---- AXIS COLORS ----
        self.ax.tick_params(axis='both', colors='#d1d4dc', labelsize=10)

        # ---- SPINES (this is what you're missing) ----
        for spine in self.ax.spines.values():
            spine.set_color('#2a2e39')
            spine.set_linewidth(1.2)

        # Axis Formatting
        self.ax.xaxis_date()

        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

        self.ax.yaxis.tick_right()
        self.ax.yaxis.set_label_position("right")

        self.ax.grid(True, alpha=0.15, color="#2a2e39")    
        # # ---- X AXIS (dates) ----
        # self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        # self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # # ---- Y AXIS (price) ----
        # self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

        # self.ax.yaxis.tick_right()
        # self.ax.yaxis.set_label_position("right")

        # self.ax.grid(True, alpha=0.15, color="#2a2e39")

    # ------------------------
    # STEP FORWARD
    # ------------------------
    def step(self):
        width=self.candle_width

        if self.i >= len(self.df):
            self.paused = True
            return

        row = self.df.iloc[self.i]
        date = self.df.index[self.i]

        o, h, l, c = row.Open, row.High, row.Low, row.Close
        color = "#26a69a" if c >= o else "#ef5350"


        self.ax.vlines(date, l, h, color=color, linewidth=1)

        self.ax.bar(
            date,
            abs(c - o),
            bottom=min(o, c),
            width=self.candle_width,
            color=color,
            alpha=0.9,
            align='center'
        )

        # markers
        if row.Action == 2:
            # BUY marker
            self.ax.scatter(date, c, color="green", marker="^", s=120, zorder=5)

            # BUY line (entry level)
            self.ax.axhline(
                y=c,
                color="green",
                linestyle="--",
                linewidth=2.5,
                alpha=0.6,
                zorder=1
            )

        elif row.Action == 0:
            # SELL marker
            self.ax.scatter(date, c, color="red", marker="v", s=120, zorder=5)

            # SELL line (exit level)
            self.ax.axhline(
                y=c,
                color="red",
                linestyle="--",
                linewidth=2.5,
                alpha=0.6,
                zorder=1
            )

        # --- CLEAN OLD SR LINES ---
        for line in self.sr_lines:
            line.remove()
        self.sr_lines.clear()

        # SR safe draw
        if row.SR_exists == 1 and not pd.isna(row.SR_price):
            line = self.ax.axhline(
                y=row.SR_price,
                color="#f0b429",
                linestyle="--",
                linewidth=1,
                alpha=0.7
            )
            self.sr_lines.append(line)

        self.i += 1   # ONLY place index advances

        # update view window
        padding = 5  # candles

        start = max(0, self.i - 50)
        end = self.i - 1

        start = max(0, start - padding)
        end = min(len(self.df) - 1, end + padding)

        # safety guard: ensure at least 2 points
        if end <= start:
            end = min(start + 1, len(self.df) - 1)

        x_min = self.df.index[start]
        x_max = self.df.index[end]

        self.ax.set_xlim(x_min, x_max)

    # ------------------------
    # RUN LOOP
    # ------------------------
    def run(self):

        plt.ion()
        self.setup()

        # draw ONLY first candle without skipping
        self.step()

        # reset index so animation starts correctly
        self.i = 1

        while self.i < len(self.df):

            if not self.paused:
                self.step()
                plt.pause(self.delay / 1000)

        plt.ioff()
        plt.show()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    # df = load_results("test_results0.85.pkl")
    # df = load_results("Dataset11_05-03_to_05-10.pkl")
    # df = load_results("RANDOMtest_results.pkl")
    df = load_results("test_results.pkl")

    engine = ReplayEngine(df, delay=200)  # speed control here
    engine.run()