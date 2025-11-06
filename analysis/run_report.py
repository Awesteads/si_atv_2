# analysis/run_report.py
import os
import shutil
import importlib
import subprocess
from datetime import datetime

def run_once(tlim, tag):
    # 1) ajusta config TIME_EXPLORER_LIMIT
    import config as cfg
    # sobrescreve o arquivo de configs dos explorers
    cfg.TIME_EXPLORER_LIMIT = tlim

    import main as main_mod
    # recria configs e roda Env via main
    # para garantir reexecução "do zero", chamamos via subprocess do próprio Python
    subprocess.run(["python", "main.py"], check=True)

    # 2) calcula stats e clustering
    subprocess.run(["python", "-m", "analysis.statistics"], check=True)
    subprocess.run(["python", "-m", "analysis.cluster_victims"], check=True)

    # 3) move outputs para pasta com tag
    out_dir = "outputs"
    tag_dir = f"outputs_{tag}"
    if os.path.exists(tag_dir):
        shutil.rmtree(tag_dir)
    shutil.copytree(out_dir, tag_dir)
    print(f"[RUN] Resultados armazenados em {tag_dir}")

def main():
    run_once(1000, "tlim1000")
    run_once(8000, "tlim8000")

if __name__ == "__main__":
    main()
