# analysis/cluster_victims.py
import os
import csv
import glob
import math
import matplotlib.pyplot as plt
from collections import defaultdict

import pandas as pd
from sklearn.cluster import KMeans

OUT_DIR = "outputs"
VICT_DS = "datasets/vict/408v/data.csv"

def load_detected_victims():
    """Lê todos os map_explorer_*.csv e retorna um DF com (victim_id,x,y) deduplicado por victim_id."""
    files = sorted(glob.glob(os.path.join(OUT_DIR, "map_explorer_*.csv")))
    if not files:
        raise FileNotFoundError("Nenhum map_explorer_*.csv encontrado em outputs/")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "victim_present" not in df.columns:
            continue
        v = df[(df["victim_present"] == 1)]
        # alguns mapas podem não ter victim_id preenchido; filtra
        if "victim_id" in v.columns:
            v = v.dropna(subset=["victim_id"])
            v["victim_id"] = v["victim_id"].astype(int)
            dfs.append(v[["victim_id", "x", "y"]])

    if not dfs:
        raise RuntimeError("Nenhuma vítima com victim_id foi encontrada nos mapas dos exploradores.")

    all_v = pd.concat(dfs, ignore_index=True)
    # mantém a primeira ocorrência por victim_id
    unique_v = all_v.drop_duplicates(subset=["victim_id"])
    return unique_v

def attach_tri_sobr(v_df):
    """Anexa tri (0-3) e sobr (0..1) usando o dataset 408v (ground truth da tarefa)."""
    ds = pd.read_csv(VICT_DS)
    ds = ds.reset_index().rename(columns={"index": "victim_id"})
    # no dataset local, victim_id é o índice (0..n-1)
    merged = v_df.merge(ds[["victim_id", "tri", "sobr"]], on="victim_id", how="left")
    return merged

def compute_kmeans(v_df, k=3, random_state=42):
    """Clusteriza por posição (x,y)."""
    X = v_df[["x", "y"]].to_numpy()
    model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = model.fit_predict(X)
    v_df = v_df.copy()
    v_df["cluster"] = labels + 1  # 1..k
    return v_df, model

def save_clusters(v_df, k=3):
    os.makedirs(OUT_DIR, exist_ok=True)
    for c in range(1, k + 1):
        part = v_df[v_df["cluster"] == c][["victim_id", "x", "y", "sobr", "tri"]]
        path = os.path.join(OUT_DIR, f"cluster{c}.txt")
        part = part.rename(columns={"victim_id": "id_vict"})
        part.to_csv(path, index=False)
        print(f"[CLUSTER] Salvo {path} com {len(part)} vítimas")

def plot_clusters(v_df, k=3):
    """Mostra os clusters na tela."""
    plt.figure(figsize=(8, 8))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for c in range(1, k + 1):
        part = v_df[v_df["cluster"] == c]
        plt.scatter(part["x"], part["y"], s=25, color=colors[c - 1], label=f"Cluster {c}", alpha=0.7)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Distribuição das vítimas por cluster")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  # <- Mostra o gráfico na tela
    print("[CLUSTER] Gráfico exibido na tela.")

def main(k=3):
    v = load_detected_victims()
    v = attach_tri_sobr(v)
    v, model = compute_kmeans(v, k=k)
    save_clusters(v, k=k)
    plot_clusters(v, k=k)  # <- Mostra visualmente
    # Resumo no console
    summary = v.groupby("cluster").agg(
        n=("victim_id", "count"),
        x_mean=("x", "mean"),
        y_mean=("y", "mean"),
        tri_mean=("tri", "mean"),
        sobr_mean=("sobr", "mean"),
    ).reset_index()
    print("\n[CLUSTER] Resumo por cluster:")
    print(summary.to_string(index=False))

    
if __name__ == "__main__":
    main(k=3)
