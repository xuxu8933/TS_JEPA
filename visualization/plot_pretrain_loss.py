import re
import os
from datetime import datetime
import matplotlib.pyplot as plt

import glob
import os

log_files = glob.glob("logs/pretrain_nike*.log")
if not log_files:
    raise FileNotFoundError("No log files found in logs/")

log_file = max(log_files, key=os.path.getctime)
print("Using log file:", log_file)


# ====== 创建 results 文件夹 ======
os.makedirs("results", exist_ok=True)

# ====== 解析日志 ======
epochs = []
losses = []
lrs = []

pattern = re.compile(
    r"Epoch\s+(\d+),\s+lr:\s*([0-9.eE+-]+)\s*-\s*JEPA Loss:\s*([0-9.eE+-]+)"
)

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            lrs.append(float(match.group(2)))
            losses.append(float(match.group(3)))

print(f"Parsed {len(epochs)} points.")

# ====== 生成时间戳 ======
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ====== 画 Loss ======
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses)
plt.xlabel("Epoch")
plt.ylabel("JEPA Loss")
plt.title("Pretrain Loss Curve")
plt.grid(True)
plt.tight_layout()

loss_path = f"results/pretrain_loss_{timestamp}.png"
plt.savefig(loss_path, dpi=200)
print("Saved loss plot to:", loss_path)

plt.show()

# ====== 画 LR（可选） ======
plt.figure(figsize=(10, 5))
plt.plot(epochs, lrs)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Curve")
plt.grid(True)
plt.tight_layout()

lr_path = f"results/pretrain_lr_{timestamp}.png"
plt.savefig(lr_path, dpi=200)
print("Saved lr plot to:", lr_path)

plt.show()