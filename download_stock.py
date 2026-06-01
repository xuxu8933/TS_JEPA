import os
import yfinance as yf

ticker = "NVDA"
start = "2015-01-01"
end = "2026-05-31"

save_dir = f"./data/{ticker}"
os.makedirs(save_dir, exist_ok=True)

df = yf.download(
    ticker,
    start=start,
    end=end,
    auto_adjust=False,
    progress=True,
)

# 如果 yfinance 返回 MultiIndex，压平成普通列
if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, "nlevels"):
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

# 保留你的模型需要的 OHLCV
df = df.reset_index()

df = df[[
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]]

df = df.dropna()

save_path = f"{save_dir}/{ticker}.csv"
df.to_csv(save_path, index=False)

print(df.head())
print("Saved to:", save_path)
print("Rows:", len(df))