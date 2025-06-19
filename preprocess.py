# preprocess.py

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Reorder 'need_maintenance' column first
    df.drop(columns=['is_deleted','api_invocation_time','write_time','eventtime','vehicle_id'], axis= 1, inplace = True)
    
    cols = ["need_maintenance"] + [c for c in df.columns if c != "need_maintenance"]
    df = df[cols]

    # 2) Ensure label is integer
    df["need_maintenance"] = df["need_maintenance"].astype(int)

    # 3) Identify non-binary columns for scaling
    non_binary_cols = [col for col in df.columns if df[col].nunique() != 2]

    # 4) Scale non-binary features
    scaler = StandardScaler()
    df[non_binary_cols] = scaler.fit_transform(df[non_binary_cols])

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-data",
        type=str,
        dest="input_data",
        required=True,
        help="S3 URI or local path of the input CSV file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        default=42,
        help="Random seed for data split repeatability"
    )
    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.input_data)

    # Clean and feature-engineer
    df = clean_and_engineer(df)

    # Initialize random generator
    rng = np.random.RandomState(args.seed)
    rand_vals = rng.rand(len(df))

    # Split into train, validation, and batch sets
    train_df = df[rand_vals < 0.8]
    val_df   = df[(rand_vals >= 0.8) & (rand_vals < 0.9)]
    batch_df = df[rand_vals >= 0.9]

    # Ensure output directories exist
    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/validation", exist_ok=True)
    os.makedirs("/opt/ml/processing/batch", exist_ok=True)

    # Write out CSVs
    train_df.to_csv("/opt/ml/processing/train/train.csv", index=False)
    val_df.to_csv("/opt/ml/processing/validation/validation.csv", index=False)
    batch_df.to_csv("/opt/ml/processing/batch/batch.csv", index=False)
