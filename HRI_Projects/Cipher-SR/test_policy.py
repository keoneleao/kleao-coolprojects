import torch
import pandas as pd
import pickle
from models import MLPPolicy
import numpy as np

def load_model(path):
    checkpoint = torch.load(path)

    model = MLPPolicy(state_dim=20, hidden_dim=96, action_dim=3)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    return model, mean, std

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


def run_policy(model, df, mean, std):
    history = []

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


    position = 0  # 0 = flat, 1 = long
    entry_price = 0
    pnl = 0

    logs = []


    last_printed_trade = -999
    # print(
    #     f"{'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} "
    #     # f"{'SR':>8} {'Fib0':>8} {'Fib100':>8} {'Action':>8}"
    #     f"{'SR':>8} {'Action':>8}"
    # )
    # print("-" * 70)



    for i in range(len(df)):

        row = df.iloc[i]
        action = 1  # default HOLD

        if i >= SEQ_LEN:

            seq = []
            for j in range(i - SEQ_LEN, i):
                row_j = df.iloc[j]
                seq.append([row_j[col] for col in state_cols])

            seq = np.array(seq).flatten()
            state = torch.tensor(seq, dtype=torch.float32)

            # --- APPLY TRAINING NORMALIZATION ---
            state = (state - mean) / std

            state = state.unsqueeze(0)

            with torch.no_grad():
                logits = model(state)

                # --- CONFIDENCE DEBUG ---
                probs = torch.softmax(logits, dim=1)
                conf, action_tensor = torch.max(probs, dim=1)

                conf = conf.item()
                raw_action = action_tensor.item()
                action = raw_action

                # --- RANDOM ACTION OVERRIDE ---
                RANDOM_MODE = True

                if RANDOM_MODE:

                    # probabilities for [SELL, HOLD, BUY]
                    random_probs = [0.05, 0.90, 0.05]

                    action = np.random.choice([0, 1, 2], p=random_probs)

                # apply filter
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

        # if i < 10:
        #     print(
        #         f"{row['Open']:8.2f} "
        #         f"{row['High']:8.2f} "
        #         f"{row['Low']:8.2f} "
        #         f"{row['Close']:8.2f} "
        #         f"{row['SR_rel']:8.2f} "
        #         # f"{row['Fib_pos']:8.2f} "
        #         # f"{row['Fib_exists']:8.2f} "
        #         f"{action:8d}"
        #     )

        # -------------------
        # LOG EVERYTHING (NO SKIPS)
        # -------------------

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

        # if action != 1 and (i - last_printed_trade > 5):

        #     print("\n=== TRADE EVENT ===")

        #     start = max(0, i - 3)
        #     end = min(len(df) - 1, i + 3)

        #     for j in range(start, end + 1):
        #         r = df.iloc[j]

        #         print(
        #             f"{r['Open']:8.2f} "
        #             f"{r['High']:8.2f} "
        #             f"{r['Low']:8.2f} "
        #             f"{r['Close']:8.2f} "
        #             f"{r['SR_rel']:8.2f} "
        #             # f"{r['Fib_pos']:8.2f} "
        #             # f"{r['Fib_exists']:8.2f} "
        #             f"{action:8d}"
        #         )

        #     print("===================\n")

        #     last_printed_trade = i

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