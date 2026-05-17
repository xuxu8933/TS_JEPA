import pandas as pd
from pathlib import Path

# Input file
input_path = Path("/home/xujiang/TS_JEPA/data/nike/NKEOrinigal.csv")

# Output file
output_path = input_path.parent / "NKENew.csv"

# Read CSV
df = pd.read_csv(input_path)

# Keep Date unchanged
date_col = "Date"

# Columns to normalize
numeric_cols = [col for col in df.columns if col != date_col]

# Z-score normalization: (x - mean) / std
df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

# Save result
df.to_csv(output_path, index=False)

print(f"Saved normalized file to: {output_path}")