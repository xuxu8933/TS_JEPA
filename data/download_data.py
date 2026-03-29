import kagglehub
import shutil
import os

src = kagglehub.dataset_download("hassanjameelahmed/nike-nke-stock-market-analysis")
dst = "./data/nike-nke-stock-market-analysis"

if os.path.exists(dst):
    shutil.rmtree(dst)

shutil.copytree(src, dst)

print("Copied to:", dst)