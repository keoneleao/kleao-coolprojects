# =========================================
# SCRIPT NAME: test_policy.py
# PURPOSE:     Loads a trained behavioral cloning model and evaluates it
#              on another loaded trading dataset. The script reconstructs
#              sequential state vectors, applies the same feature
#              normalization used during training, predicts trading
#              actions, applies trading constraints, and saves the
#              resulting actions for replay and evaluation.
# AUTHOR:      Keone Leao
# DATE:        04/21/26
# DEPENDENCIES:torch, torch, pandas, pickle, MLPPolicy, Numpy
# =========================================

## Imports
import torch
import pandas as pd
import pickle
from models import MLPPolicy
import numpy as np

# load model & additionally weights, means, standard deviations
def load_model(path):
    checkpoint = torch.load(path)

    model = MLPPolicy(state_dim=20, hidden_dim=96, action_dim=3)
    model.load_state_dict(checkpoint["model"])
    model.eval() # tells PyTorch, modeling is over & disables Dropout

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    return model, mean, std

#load dataset while preserving replay information
def load_dataset(path):
    with open(path, "rb") as f:
        df = pickle.load(f)

        feature_cols = [
            "log_close",
            "return_1",
            "return_3",
            "SR_rel",
            "SR_exists",
            "Fib_pos",
            "Fib_exists",
            "High",
            "Low",
            "Open",
            "Close",
            "SR_price",
            "Fib_y1",
            "Fib_y2"
        ]

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return df

# Execute trained policy on loaded dataset
# construct a dataset off this
def run_policy(model, df, mean, std):
    history = [] # consider deleteing: never used

    SEQ_LEN = 4
    state_cols = [
        "log_close",
        "return_1",
        "return_3",
        "SR_rel",
        "SR_exists",
        # "Fib_pos",
        # "Fib_exists"
    ]

    # variables help model during training
    position = 0  # 0 = flat, 1 = long # in position?
    entry_price = 0  # entry position
    pnl = 0 # profit/loss

    logs = []


    last_printed_trade = -999
    # print(
    #     f"{'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} "
    #     # f"{'SR':>8} {'Fib0':>8} {'Fib100':>8} {'Action':>8}"
    #     f"{'SR':>8} {'Action':>8}"
    # )
    # print("-" * 70)


    # main training loop
    for i in range(len(df)):

        row = df.iloc[i]
        action = 1  # default HOLD

        if i >= SEQ_LEN: # wait until you have 4 previous states to start making decisions

            seq = [] # put previous 4 states into 20 dimensional input
            for j in range(i - SEQ_LEN, i):
                row_j = df.iloc[j]
                seq.append([row_j[col] for col in state_cols])

            seq = np.array(seq).flatten()
            state = torch.tensor(seq, dtype=torch.float32)

            # --- APPLY TRAINING NORMALIZATION ---
            state = (state - mean) / std

            state = state.unsqueeze(0) # convert 20 into 1x20

            with torch.no_grad(): # disables gradient computation during inference for speed and less memory
                logits = model(state)

                # --- CONFIDENCE DEBUG ---
                probs = torch.softmax(logits, dim=1)
                conf, action_tensor = torch.max(probs, dim=1)

                conf = conf.item()
                raw_action = action_tensor.item()
                action = raw_action

                # --- RANDOM ACTION OVERRIDE ---
                RANDOM_MODE = False # True: actions are random instead of using trained model

                if RANDOM_MODE:

                    # probabilities for [SELL, HOLD, BUY]
                    random_probs = [0.05, 0.90, 0.05]

                    action = np.random.choice([0, 1, 2], p=random_probs)

                # apply filter (do not buy/sell unless confident enough)
                if action != 1 and conf < 0.85:
                    action = 1

                # --- DO NOT BUY ABOVE SR LINE ---
                if action == 2:  # BUY

                    sr_price = row["SR_price"]
                    close_price = row["Close"]

                    # skip if no SR exists
                    if row["SR_exists"] == 1:

                        # if price is ABOVE SR, cancel buy
                        if close_price > sr_price:
                            action = 1  # HOLD

                # --- POSITION CONSTRAINTS ---

                price = row["Close"]

                # If trying to SELL but not profitable → block it
                if action == 0:  # SELL
                    if position == 0:
                        action = 1  # can't sell if not holding
                    elif price < entry_price * 1.0: # increase scaling factor to ensure profit
                        action = 1  # don't sell at a loss

                # If trying to BUY but already long → block it
                if action == 2:  # BUY
                    if position == 1:
                        action = 1  # already holding

                # --- UPDATE POSITION STATE ---
                if action == 2:  # BUY
                    position = 1
                    entry_price = price

                elif action == 0:  # SELL
                    position = 0
                    entry_price = 0

                print(f"Probs: {probs.numpy()} | Conf: {conf:.3f} | Raw: {raw_action} | Final: {action}")

        

        record = {
            "i": i,
            "price": row["Close"],
            "action": action
        }

        history.append(record)

        logs.append([
            row["log_close"],
            row["return_1"],
            row["return_3"],
            row["SR_rel"],
            row["SR_exists"],
            row["Fib_pos"],
            row["Fib_exists"],
            action,
            df.index[i],
            row["High"],
            row["Low"],
            row["Open"],
            row["Close"],
            row["SR_price"],
            row["Fib_y1"],
            row["Fib_y2"]
        ])


    columns = [
        "log_close",
        "return_1",
        "return_3",
        "SR_rel",
        "SR_exists",
        "Fib_pos",
        "Fib_exists",
        "Action",
        "Time",
        "High",
        "Low",
        "Open",
        "Close",
        "SR_price",
        "Fib_y1",
        "Fib_y2"
    ]

    return pd.DataFrame(logs, columns=columns)

if __name__ == "__main__":

    model, mean, std = load_model("model_weightsparm")
    df = load_dataset("Dataset11_05-03_to_05-10.pkl")

    results = run_policy(model, df, mean, std)

    # --- SAVE FILES ---

    # JSON (human-readable)
    results.to_json(
        "test_results.json",
        orient="records",
        lines=True,
        date_format="iso"
    )

    # PKL (fast for training later)
    with open("test_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Saved test results to JSON and PKL")