import os
import pandas as pd
import numpy as np

src_csv = "/Users/yuwei/TS_JEPA/data/nike-nke-stock-market-analysis/NKE.csv"

df = pd.read_csv(src_csv)

df["date"] = pd.to_datetime(df["Date"], errors="coerce")

df["open_r"] = np.log(pd.to_numeric(df["Open"], errors="coerce")).diff()
df["high_r"] = np.log(pd.to_numeric(df["High"], errors="coerce")).diff()
df["low_r"] = np.log(pd.to_numeric(df["Low"], errors="coerce")).diff()
df["close_r"] = np.log(pd.to_numeric(df["Close"], errors="coerce")).diff()
df["volume_r"] = np.log(pd.to_numeric(df["Volume"], errors="coerce") + 1).diff()

df = df.dropna().sort_values("date").reset_index(drop=True)

features = ["open_r", "high_r", "low_r", "close_r", "volume_r"]

for col in features:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std

out_dir = "./data/nike"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "nike.csv")

df[["date"] + features].to_csv(out_path, index=False)

print("Saved to:", out_path)
print(df[["date"] + features].head())
print(df[["date"] + features].columns.tolist())