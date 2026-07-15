# =========================================
# SCRIPT NAME: get_stockdataset.py
# PURPOSE:     Creates a GUI for user to trade on using real stock data. Saves decisions and important info for training. 
#              Place support/resistance lines, fib retracements, and buy/sell trades.
# AUTHOR:      Keone Leao
# DATE:        04/21/26
# DEPENDENCIES:matplotlib, matplotlib.pyplot, matplotlib.dates, yfinance, pandas, pickle, numpy, datetime
# =========================================

## Imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
SYMBOL    = 'TSLA'
#           'year-mo-dy'
START     = '2026-03-23'
END       = '2026-03-26'
start_obj = datetime.strptime(START, "%Y-%m-%d")
end_obj = datetime.strptime(END, "%Y-%m-%d")

DATASET_BASENAME = f"Delete#_{start_obj.strftime('%m-%d')}_to_{end_obj.strftime('%m-%d')}"
INTERVAL  = '5m'   # examples: '1m', '5m', '15m', '1h', '1d'
# each interval has a maximum duration. 1m: 7 days, 5m-30m: 60 days, 1h: 730 days
# ─────────────────────────────────────────────────────────────────────────────

# Global Variables
sr_levels = []   # store prices only
sr_lines = []    # ONLY for drawn matplotlib objects
fib_points = []
fib_artists = []
trade_lines = []
trade_markers = []
trades = []  # store structured trade info
fib_active = False
fib_y1 = None
fib_y2 = None
mode = "sr"   # "sr" = support/resistance, "fib" = fibonacci

current_idx = 1 # candle index

# Functions

# Download historical OHLC data and standardize the dataframe format
def download_data(symbol, start, end, interval):
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)   # flatten yfinance 0.2+ columns
    df.index = pd.to_datetime(df.index)
    return df

# render candlestick bodies and wicks using Matplotlib
def draw_candles(ax, df):
    dates = mdates.date2num(df.index.to_pydatetime())
    # compute spacing between candles
    if len(dates) > 1:
        delta = dates[1] - dates[0]
    else:
        delta = 1  # fallback

    w = delta * 0.8   # 80% of spacing looks good

    for i, (date, row) in enumerate(zip(dates, df.itertuples())):
        o, h, l, c = row.Open, row.High, row.Low, row.Close
        color = '#26a69a' if c >= o else '#ef5350'   # green up / red down

        # High-low wick
        ax.plot([date, date], [l, h], color=color, linewidth=0.8, zorder=2)
        # Open-close body
        body_bottom = min(o, c)
        body_height = abs(c - o) or 0.01   # avoid zero-height
        rect = plt.Rectangle(
            (date - w / 2, body_bottom), w, body_height,
            facecolor=color, edgecolor=color, linewidth=0.5, zorder=3
        )
        ax.add_patch(rect)

# removes every dran candle
def clear_candles(ax):
    # remove line collections (wicks)
    for coll in ax.collections[:]:
        coll.remove()

    # remove candle bodies (rectangles)
    for patch in ax.patches[:]:
        patch.remove()

# reveals one candle at a time
def draw_next_candle(ax, df, i):
    dates = mdates.date2num(df.index.to_pydatetime())

    # compute width
    if len(dates) > 1:
        delta = dates[1] - dates[0]
    else:
        delta = 1

    w = delta * 0.8

    row = df.iloc[i]
    date = dates[i]

    o, h, l, c = row.Open, row.High, row.Low, row.Close
    color = '#26a69a' if c >= o else '#ef5350'

    # wick
    ax.plot([date, date], [l, h], color=color, linewidth=0.8, zorder=2)

    # body
    body_bottom = min(o, c)
    body_height = abs(c - o) or 0.01

    rect = plt.Rectangle(
        (date - w / 2, body_bottom),
        w,
        body_height,
        facecolor=color,
        edgecolor=color,
        linewidth=0.5,
        zorder=3
    )
    ax.add_patch(rect)

# only draw candes 0-n
def render_candles(ax, df, n):

    dates = mdates.date2num(df.index.to_pydatetime())
    # compute spacing between candles
    if len(dates) > 1:
        delta = dates[1] - dates[0]
    else:
        delta = 1  # fallback

    w = delta * 0.8   # 80% of spacing looks good

    subset = df.iloc[:n]

    for date, row in zip(dates[:n], subset.itertuples()):
        o, h, l, c = row.Open, row.High, row.Low, row.Close
        color = '#26a69a' if c >= o else '#ef5350'

        ax.plot([date, date], [l, h], color=color, linewidth=0.8, zorder=2)

        body_bottom = min(o, c)
        body_height = abs(c - o) or 0.01

        rect = plt.Rectangle(
            (date - w / 2, body_bottom),
            w,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
            zorder=3
        )
        ax.add_patch(rect)

# set up GUI axies
def setup_axes(ax, df, symbol):
    ax.set_facecolor('#131722')
    ax.set_ylabel('Price (USD)', color='#d1d4dc', fontsize=11)
    ax.tick_params(colors='#d1d4dc')
    ax.set_title(f'{symbol} — click to add support/resistance lines',
                 color='#d1d4dc', fontsize=13, pad=10)

    # X-axis: readable dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', color='#d1d4dc')

    # Y-axis limits with padding
    pad = (df['High'].max() - df['Low'].min()) * 0.05
    ax.set_ylim(df['Low'].min() - pad, df['High'].max() + pad)

    dates = mdates.date2num(df.index.to_pydatetime())
    ax.set_xlim(dates[0] - 1, dates[-1] + 1)

    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2e39')
    ax.yaxis.label.set_color('#d1d4dc')
    ax.grid(axis='y', color='#2a2e39', linewidth=0.5, linestyle='--')

# consider deleting: not used later
def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


# mouse click snaps to nearest visible candle price of OHLC
def snap_to_candle(df, x, y):
    global current_idx
    dates = mdates.date2num(df.index.to_pydatetime())

    # restrict to visible candles only
    visible_dates = dates[:current_idx]

    idx = (abs(visible_dates - x)).argmin()

    row = df.iloc[idx]

    prices = {
        "High": row.High,
        "Low": row.Low,
        "Open": row.Open,
        "Close": row.Close
    }

    # Snap to closest price
    closest_label = min(prices, key=lambda k: abs(prices[k] - y))
    snapped_price = prices[closest_label]

    return idx, closest_label, snapped_price

# compute normalized current price to neareset SR level for training
def update_sr_state_for_candle(i):
    if len(sr_levels) == 0:
        return

    close_price = state_df["Close"].iloc[i]

    sr_array = pd.Series(sr_levels)

    nearest_sr = sr_array.iloc[
        (sr_array - close_price).abs().argmin()
    ]

    if close_price == 0 or np.isnan(close_price):
        sr_norm = 0
    else:
        sr_norm = (nearest_sr - close_price) / close_price

    state_df.iat[i, state_df.columns.get_loc("SR_rel")] = sr_norm
    state_df.iat[i, state_df.columns.get_loc("SR_exists")] = 1
    state_df.iat[i, state_df.columns.get_loc("SR_price")] = nearest_sr

# returns relative position inside fib retracement (between 0-1)
def update_fib_state_for_candle(i):

    # skip ONLY if user is currently placing fib points
    if len(fib_points) != 0:
        return

    # must have an active fib drawn
    if not fib_active:
        return

    close_price = state_df["Close"].iloc[i]

    y1 = fib_y1
    y2 = fib_y2

    y_low = min(y1, y2)
    y_high = max(y1, y2)

    if y_high - y_low == 0 or np.isnan(close_price):
        fib_pos = 0
    else:
        fib_pos = (close_price - y_low) / (y_high - y_low)

    fib_pos = np.clip(fib_pos, 0.0, 1.0)

    state_df.iat[i, state_df.columns.get_loc("Fib_pos")] = fib_pos
    state_df.iat[i, state_df.columns.get_loc("Fib_exists")] = 1
    state_df.iat[i, state_df.columns.get_loc("Fib_y1")] = y1
    state_df.iat[i, state_df.columns.get_loc("Fib_y2")] = y2

# Handles keyboard shortcuts
def onkey(event):
    global mode, fig, current_idx

    print(f"Key pressed: {event.key}") # lets you know what key has been pressed

    if event.key == 'f': # clicking makes a fib
        mode = "fib"
        print("Switched to Fibonacci tool")

    elif event.key == 'd': # clicking makes a SR line
        mode = "sr"
        print("Switched to Support/Resistance tool")

        # Optional: clear unfinished fib
        #fib_points.clear()

    elif event.key == 'e': # erases last SR line
        print("Removing last SR line")

        if len(sr_levels) == 0:
            print("No SR lines to remove")
            return

        # remove last SR level
        sr_levels.pop()

        # remove last 2 drawn objects (line + text)
        if len(sr_lines) >= 2:
            for _ in range(2):
                item = sr_lines.pop()
                try:
                    item.remove()
                except ValueError:
                    pass

        # --- CLEAR STATE FOR FUTURE CANDLES ---
        start_idx = max(current_idx - 1, 0)

        state_df.loc[state_df.index[start_idx]:, "SR_rel"] = float("nan")
        state_df.loc[state_df.index[start_idx]:, "SR_exists"] = 0
        state_df.loc[state_df.index[start_idx]:, "SR_price"] = float("nan")

        # --- REBUILD SR STATE FROM REMAINING LEVELS ---
        for i in range(start_idx, len(state_df)):
            update_sr_state_for_candle(i)
        

    elif event.key == ' ': # space reveals next candle

        if current_idx >= len(df):
            print("End of dataset reached")
            return

        draw_next_candle(ax, df, current_idx)

        # update SR
        update_sr_state_for_candle(current_idx)

        # 🔥 update Fibonacci
        update_fib_state_for_candle(current_idx)

        current_idx += 1

    elif event.key == 't': # clicking now buys/sells
        mode = "trade"
        print("Switched to Buy/Sell tool")

    elif event.key == 'c': # clears trades
        print("Clearing trades")

        for item in trade_lines + trade_markers:
            try:
                item.remove()
            except ValueError:
                pass

        trade_lines.clear()
        trade_markers.clear()
        trades.clear()

    elif event.key == 'p': # print an update of visible candes to terminal
        print("\nSTATE SNAPSHOT (visible candles):\n")

        visible_df = state_df.iloc[:current_idx]

        print(visible_df.tail(5).to_string())

    elif event.key == 'x': # disable clicking input in GUI
        mode = "none"
        print("Mouse input disabled")

    fig.canvas.draw_idle()

# draws fib
def draw_fib(ax, p1, p2):
    (x1, y1), (x2, y2) = p1, p2

    # Fibonacci levels
    levels = {
        "0.0%": 0.0,
        "23.6%": 0.236,
        "38.2%": 0.382,
        "50.0%": 0.5,
        "61.8%": 0.618,
        "78.6%": 0.786,
        "100.0%": 1.0
    }

    artists = []

    for label, lvl in levels.items():
        y = y1 + (y2 - y1) * lvl

        # Draw horizontal line
        line = ax.axhline(
            y=y,
            linestyle='--',
            linewidth=1,
            alpha=0.7,
            color='#42a5f5',
            zorder=4
        )

        # Draw label on the right side (like your SR lines)
        text = ax.text(
            1.01, y,
            f'{label}  (${y:,.2f})',
            transform=ax.get_yaxis_transform(),
            fontsize=8,
            color='#42a5f5',
            va='center',
            clip_on=False
        )

        artists.append(line)
        artists.append(text)

    return artists

# removes fib and recorded state
def clear_fib():
    global fib_active
    fib_active = False
    global fib_artists

    for artist in fib_artists:
        try:
            artist.remove()
        except ValueError:
            pass

    fib_artists.clear()

    # ALSO clear state going forward
    start_idx = max(current_idx - 1, 0)

    state_df.loc[state_df.index[start_idx]:, "Fib_pos"] = float("nan")
    state_df.loc[state_df.index[start_idx]:, "Fib_exists"] = 0
    state_df.loc[state_df.index[start_idx]:, "Fib_y1"] = float("nan")
    state_df.loc[state_df.index[start_idx]:, "Fib_y2"] = float("nan")
    state_df.loc[state_df.index[start_idx]:, "Fib_exists"] = 0

# clicking does different things based on keyboard shortcut mode
def onclick(event):
    if event.inaxes != ax:
        return

    global fib_points, fib_artists, sr_levels, sr_lines, mode

    x, y = event.xdata, event.ydata

    if mode == "none":
        return

    if mode == "sr":
        sr_levels.append(y)

        sr_array = pd.Series(sr_levels)

        i = max(current_idx - 1, 0)

        close_price = state_df["Close"].iloc[i]

        nearest_sr = sr_array.iloc[
            (sr_array - close_price).abs().argmin()
        ]

        if close_price == 0 or np.isnan(close_price):
            sr_norm = 0
        else:
            sr_norm = (nearest_sr - close_price) / close_price

        state_df.iat[i, state_df.columns.get_loc("SR_rel")] = sr_norm
        state_df.iat[i, state_df.columns.get_loc("SR_exists")] = 1
        state_df.iat[i, state_df.columns.get_loc("SR_price")] = nearest_sr

        line = ax.axhline(
            y=y,
            color='#f0b429',
            linestyle='--',
            linewidth=1.2,
            alpha=0.85,
            zorder=5
        )

        text = ax.text(
            1.01, y,
            f'  ${y:,.2f}',
            transform=ax.get_yaxis_transform(),
            color='#f0b429',
            fontsize=8,
            va='center',
            clip_on=False
        )

        sr_lines.append(line)
        sr_lines.append(text)

    elif mode == "fib":
        if len(fib_points) == 0:
            clear_fib()

        fib_points.append((x, y))

        if len(fib_points) == 2:
            p1, p2 = fib_points[0], fib_points[1]

            clear_fib()
            fib_artists = draw_fib(ax, p1, p2)

            # --- STORE STATE ---
            y1 = p1[1]
            y2 = p2[1]

            # store globally
            global fib_active, fib_y1, fib_y2
            fib_active = True
            fib_y1 = y1
            fib_y2 = y2

            # compute for ALL candles going forward
            for i in range(len(state_df)):
                update_fib_state_for_candle(i)

            fib_points.clear()

    elif mode == "trade":
        idx, label, price = snap_to_candle(df, x, y)
        row = df.iloc[idx]

        # --- Determine BUY/SELL via modifier keys ---
        if event.button == 1:   # left click
            tag = "BUY"
            color = '#26a69a'
        elif event.button == 3: # right click
            tag = "SELL"
            color = '#ef5350'
        else:
            print("Buy: left-click, Sell: right-click")
            return

        dates = mdates.date2num(df.index.to_pydatetime())
        candle_x = dates[idx]

        # --- Draw horizontal line ---
        line = ax.axhline(
            y=price,
            color=color,
            linestyle='-',
            linewidth=1.5,
            alpha=0.9,
            zorder=6
        )

        # --- Label ---
        text = ax.text(
            1.01, price,
            f'{tag} {label}  (${price:,.2f})',
            transform=ax.get_yaxis_transform(),
            color=color,
            fontsize=8,
            va='center',
            clip_on=False
        )

        # --- Marker (dot on candle) ---
        marker, = ax.plot(
            candle_x, price,
            marker='o',
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=0.8,
            color=color,
            zorder=10
        )

        # --- Store everything ---
        timestamp = state_df.index[idx]

        if tag == "BUY":
            state_df.at[timestamp, "Action"] = 2
        elif tag == "SELL":
            state_df.at[timestamp, "Action"] = 0
        trade_lines.append(line)
        trade_lines.append(text)
        trade_markers.append(marker)

        trades.append({
            "type": tag,
            "price": price,
            "candle_index": idx,
            "label": label
        })

    fig.canvas.draw_idle()


# ── MAIN ──────────────────────────────────────────────────────────────────────
# Setup and draw the plot
print(f"Downloading {SYMBOL}...")
df = download_data(SYMBOL, START, END, INTERVAL)

# --- State ---
state_df = pd.DataFrame(index=df.index)
base_price = df["Close"].iloc[0]

# --- RAW DATA (for replay only) ---
state_df["Time"] = df.index
state_df["Open"] = df["Open"]
state_df["High"] = df["High"] 
state_df["Low"] = df["Low"]
state_df["Close"] = df["Close"]

state_df["SR_price"] = float("nan")

state_df["Fib_y1"] = float("nan")
state_df["Fib_y2"] = float("nan")

# --- CORE STATE ---
# relative features (what model learns from)
state_df["log_close"] = np.log(df["Close"] / base_price)
# --- temporal features ---
state_df["return_1"] = df["Close"].pct_change(1)
state_df["return_3"] = df["Close"].pct_change(3)

# fill first NaN values
state_df["return_1"] = state_df["return_1"].fillna(0)
state_df["return_3"] = state_df["return_3"].fillna(0)
state_df["SR_rel"] = float("nan")
state_df["SR_exists"] = 0
state_df["Fib_exists"] = 0
state_df["Fib_pos"] = float("nan")

# action (label)
state_df["Action"] = 1  # 0=sell, 1=hold, 2=buy

# -------------

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#131722')

# draw_candles(ax, df) # draws all candles at once
render_candles(ax, df, current_idx) # one candle at a time
setup_axes(ax, df, SYMBOL)

# Event handlers: "hey Matplotlib if this happens, call this funciton"
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)

# make sure figure is the right size
plt.tight_layout()
plt.subplots_adjust(right=0.94)

# Open window for Figure
plt.show() # loop that keeps the window open and listens for events
# everything after plt.show() waits until the window closes!

# --- FINAL COLUMN ORDER (VERY IMPORTANT) ---
state_df = state_df[
    [
        "log_close",
        "return_1",
        "return_3",
        "SR_rel",
        "SR_exists",
        "Fib_pos",
        "Fib_exists",
        "Action",

        # ---- replay metadata ----
        "Time",
        "High",
        "Low",
        "Open",
        "Close",
        "SR_price",
        "Fib_y1",
        "Fib_y2"
    ]
]

# --- SAVE STATE AFTER WINDOW CLOSES ---

json_name = f"{DATASET_BASENAME}.json"
pkl_name = f"{DATASET_BASENAME}.pkl"

state_df.to_json(
    json_name,
    orient="records",
    lines=True,
    date_format="iso"
)

with open(pkl_name, "wb") as f:
    pickle.dump(state_df, f)

print(f"State saved to {pkl_name}")

# Done:
# 1. be able to drag around a fibonacii table
# 2. have candles appear one at a time
# 3. have code log the data (state, action)
# 4. Make model
# 5. Make training policy and have it save to a pkl and json (for viewing)
# 6. Make a testing policy that decides when to buy/sell off of state and model
# 7. Make replay code that replays in state dataset (including get_stockdata and test data)
#       Have this increment like stuff is happening in real time in a GUI
# 8. Change stat: only look at close, take in SR Line - close, take in fib start - close, take in fib end - fib close
# 9. Change training weights (should consider HOLD much less than it does to accomidate for its frequency)

# Problem Statement
# - Day trading takes time because the monitor needs to be monitored to buy and sell at the right levels, trading can passive to maximize income
# - The human will use the strategy it want the robot to execute and let the robot buy and sell without having to monitor it
# - This will free the human to do other things such as have a job while still trading, elliminating the need for an investment manager
# - formulas for imitation learning
# - investment manager or stop/buy orders, these do the whole service without input (takes a percentage), or is rigid things you set without getting more info from the market (does not include price info after order is placed)
# - These approaches don't allow you to use your strategy with your rational while constantly monitoring the prices to get in and out at the right time

# Method
# - I propose an imitation model that will learn off demonstration you set. All you need to due is provide some context for your strategy and the model will try to imitate it
# - equations for machine learning, state/action vector, process of training to testing to replay (at the end you can extend it to more real world stuff if you like what you see in testing)
# - the model is not trained to optimize trades but just try to replicate what you would do in your place (this is a strength for not doing anything crazy, and a weakness of possible loosing gains)
# - find some existing alternative and think about how to compare performance (its possible that nothing does what we do)

# Results
# - train model, test it on a dataset different from the training dataset, then trade yourself using get_dataset, then replay them both side by side and calculate error between them
# - We want to see if the model will replicate your decisions, the best way to do that is to trade the exact same dataset as the model and see if yall do the same thing
# - include any baselines? (its possible that nothing does what we do)
# - calculate error between two models, talk about the size of the error and what it means for usability and accuracy (is it usable vs is it accurate vs is it consistent with error)
# - surprising results?

# Connection to Class
# - imitation learning, what makes a good state/action vector, adjusting learning parameters (architecture and parameters) and demonstrations
# - lecture constantly talks about physical robots and humans interacting, but what if the robot in this circumstance is intangible and is trained off stocks which are seemingly random. I'm also making a state vector completely different from what we've done, 
# - and different testing environments. demonstrations can seam random as you may miss the perfect enter spot cuz you didnt have enough info but the robot doesn't know that 

# Challenges
# - the project was pretty challenging while not straying too far from concepts and practices utilized in class while adding an element of unfamilularity that added challenges that reinforce topics learned in class
# 1. Creating the GUI and making sure it did as intended [think about what the model needs to learn]
# 2. Creating state vector and deciding what should be in there (got changed later) [think about what makes a state, what is useful for a robot, how the robot will interpret a state, normalization to generalize what the robot learns]
# 3. Changing the train policy to not look at HOLD so much [how to deal with rarely changing states and sparse actions]
# 4. Creating a whole test code for my specific circumstances []
# 5. Making the replay GUI? (this one wan't really a challenge) [didn't really learn anything but code strucutre, might take the challenge out]
# 6. Getting data, I need to get multiple datasets that are good datasets (what is a good dataset?) []

