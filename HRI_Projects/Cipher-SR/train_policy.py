import torch
import torch.nn as nn
import pickle
from models import MLPPolicy
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


def build_state(row):
    return [
        row["log_close"],
        row["SR_rel"],
        row["SR_exists"],
        row["Fib_pos"],
        row["Fib_exists"]
    ]

# import dataset for training
class MyData(Dataset):

    def __init__(self, loadname):

        with open(loadname, "rb") as f:
            df = pickle.load(f)

        # ---- DEFINE EXACT STATE VECTOR ----
        state_cols = [
            "log_close",
            "return_1",
            "return_3",
            "SR_rel",
            "SR_exists",
        ]

        action_col = "Action"

        # drop NaNs / cleanup
        df = df[state_cols + [action_col, "dataset_id"]].copy()
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # ---- Downsample HOLD ----
        dfs = []

        for dataset_id, group in df.groupby("dataset_id"):

            hold_mask = group[action_col] == 1
            non_hold_mask = group[action_col] != 1

            hold_df = group[hold_mask]
            non_hold_df = group[non_hold_mask]

            hold_df = hold_df.iloc[::3]

            new_group = pd.concat([hold_df, non_hold_df]).sort_index()
            dfs.append(new_group)

        df = pd.concat(dfs).reset_index(drop=True)

        SEQ_LEN = 4  # number of timesteps

        states_seq = []
        actions_seq = []

        SEQ_LEN = 4

        for dataset_id, group in df.groupby("dataset_id"):

            data = group[state_cols].values
            actions = group[action_col].values

            for i in range(SEQ_LEN, len(data)):
                seq = data[i-SEQ_LEN:i].flatten()
                states_seq.append(seq)
                actions_seq.append(actions[i])

        self.states = torch.tensor(np.array(states_seq), dtype=torch.float32)
        # normalize features (VERY IMPORTANT for stability)
        self.mean = self.states.mean(dim=0)
        self.std = self.states.std(dim=0) + 1e-6

        # --- APPLY NORMALIZATION ---
        self.states = (self.states - self.mean) / self.std

        self.actions = torch.tensor(actions_seq, dtype=torch.long)

        print("imported dataset length:", len(self.states))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# train model
def train_model(loadname):

    # training parameters
    print("[-] training bc")
    EPOCH = 80
    LR = 5e-4

    # initialize model and optimizer
    model = MLPPolicy(state_dim=20, hidden_dim=96, action_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # initialize dataset
    print("[-] loading data: " + loadname)
    train_data = MyData(loadname)
    BATCH_SIZE = 128 # 64 or 128
    print("my batch size is:", BATCH_SIZE)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # BEFORE training loop
    all_actions = train_data.actions.long()
    unique, counts_np = np.unique(all_actions.numpy(), return_counts=True)
    print("Action distribution:", dict(zip(unique, counts_np)))

    counts = torch.bincount(all_actions)

    # inverse frequency
    weights = 1.0 / (counts.float() + 1e-6)

    # normalize so average weight ≈ 1
    weights = weights / weights.mean()

    # OPTIONAL: manually reduce HOLD importance
    # assuming HOLD = 1
    weights[1] = weights[1] * 0.5  # reduce HOLD weight (good range: 0.05 – 0.2) [if too low, model will spam buys/sells]

    print("Class weights:", weights)

    criterion = nn.CrossEntropyLoss(weight=weights)

    # main training loop
    for epoch in range(EPOCH + 1):
        for states, actions in train_set:

            actions_hat = model(states)

            loss = criterion(actions_hat, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch % 10 == 0:
            print(epoch, loss.item())
            preds = torch.argmax(actions_hat, dim=1)
            print("Pred distribution:", torch.bincount(preds))
            torch.save({
                "model": model.state_dict(),
                "mean": train_data.mean,
                "std": train_data.std
            }, "model_weightsparm")

# train models
if __name__ == "__main__":
    train_model("CombinedDataset1to10_03-08_to_05-02.pkl")