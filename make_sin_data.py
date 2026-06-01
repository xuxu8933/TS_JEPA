import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

N = 3000
t = np.arange(N)

# 基础周期信号
signal = np.sin(2 * np.pi * t / 50)

# 随机噪声
noise = np.random.normal(
    loc=0.0,
    scale=0.15,
    size=N
)

# 模拟价格：保持为正数
close = 10 + signal + noise

df = pd.DataFrame({
    "Date": pd.date_range("2020-01-01", periods=N, freq="D"),
    "Close": close
})

df.to_csv("sin_noise_data.csv", index=False)

plt.figure(figsize=(12, 4))
plt.plot(df["Date"][:300], df["Close"][:300])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Noisy Sin Data for Training")
plt.grid(True)
plt.tight_layout()
plt.show()