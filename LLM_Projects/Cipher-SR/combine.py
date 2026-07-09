import pandas as pd
import pickle

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(df, path):
    with open(path, "wb") as f:
        pickle.dump(df, f)

def combine_datasets(file_list, output_path):

    dfs = []

    expected_cols = [
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

    for file in file_list:
        print(f"Loading {file}")
        df = load_pkl(file)

        # --- Ensure consistent columns ---
        df = df.reindex(columns=expected_cols)

        # --- Clean ---
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # --- Reset index ---
        df = df.reset_index(drop=True)

        # --- Dataset ID ---
        df["dataset_id"] = file.split(".")[0]

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # --- Shuffle ---
    # combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- SAVE PKL ---
    save_pkl(combined_df, output_path)
    print(f"Saved PKL to {output_path}")

    # --- SAVE JSON ---
    json_path = output_path.replace(".pkl", ".json")

    combined_df.to_json(
        json_path,
        orient="records",
        lines=True,
        date_format="iso"
    )

    print(f"Saved JSON to {json_path}")


if __name__ == "__main__":

    files = [
        "Dataset1_03-19_to_03-21.pkl",
        "Dataset2_03-23_to_03-26.pkl",
        "Dataset3_04-01_to_04-05.pkl",
        "Dataset4_04-07_to_04-12.pkl",
        "Dataset5_04-14_to_04-16.pkl",
        "Dataset6_04-18_to_04-21.pkl",
        "Dataset7_04-22_to_04-26.pkl",
        "Dataset8_04-27_to_05-02.pkl",
        "Dataset9_03-08_to_03-14.pkl",
        "Dataset10_03-15_to_03-19.pkl"
    ]

    combine_datasets(files, "Delete_CombinedDataset1to10_03-08_to_05-02.pkl")

    # ----------------------------
    # VALIDATION (ADD THIS PART)
    # ----------------------------
    print("\n--- VALIDATING COMBINED DATASET ---")

    df = load_pkl("Delete_CombinedDataset1to10_03-08_to_05-02.pkl")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nHead:")
    print(df.head(10))

    print("\nTail:")
    print(df.tail(10))

    print("\nDescribe:")
    print(df.describe())

    print("\nNaN counts:")
    print(df.isna().sum())

    print("\nAction distribution:")
    print(df["Action"].value_counts())

    print("\nSR_exists distribution:")
    print(df["SR_exists"].value_counts())

    print("\nFib_exists distribution:")
    print(df["Fib_exists"].value_counts())