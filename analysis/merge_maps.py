# analysis/merge_maps.py
import os
import csv
from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 🟢 ADIÇÃO

OUTPUT_DIR = "outputs"
UNIFIED_FILE = os.path.join(OUTPUT_DIR, "map_unificado.txt")  # 🔄 volta pro .txt (compatível com visualizer)

# prioridade de status (para resolver conflitos)
STATUS_PRIORITY = {"wall": 3, "clear": 2, "unknown": 1, "out_of_bounds": 0}


def read_agent_map(filepath):
    """Lê o CSV de um agente e retorna {(x, y): dict(row)}"""
    grid = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y = int(row["x"]), int(row["y"])
            grid[(x, y)] = row
    return grid


def unify_maps(maps):
    """Une múltiplos mapas de agentes em um só, resolvendo conflitos"""
    unified = {}

    all_coords = set()
    for grid in maps:
        all_coords |= set(grid.keys())

    for xy in all_coords:
        cells = [g[xy] for g in maps if xy in g]

        # status com maior prioridade
        chosen_status = max(
            (c["status"] for c in cells),
            key=lambda s: STATUS_PRIORITY.get(s, 0),
        )

        # média dos floor_factors conhecidos
        floors = [
            float(c["floor_factor"])
            for c in cells
            if c["floor_factor"] not in ("", "None")
        ]
        avg_floor = mean(floors) if floors else None

        # vítima se qualquer um detectou
        victim_present = any(int(c["victim_present"]) for c in cells)

        # 🟢 ADIÇÃO: triagem inicial None
        tri_value = None

        # 🟢 ADIÇÃO: tenta pegar tri do dataset se houver vítima
        if victim_present:
            # tenta capturar o victim_id se existir no CSV do agente
            victim_ids = [
                int(c["victim_id"]) for c in cells if "victim_id" in c and c["victim_id"].isdigit()
            ]
            if victim_ids:
                victim_id = victim_ids[0]
                tri_value = get_tri_from_dataset(victim_id)

        unified[xy] = {
            "x": xy[0],
            "y": xy[1],
            "status": chosen_status,
            "floor_factor": avg_floor,
            "victim_present": int(victim_present),
            "tri": tri_value,  # 🟢 ADIÇÃO
        }

    return unified


# 🟢 ADIÇÃO: função para buscar tri no dataset de vítimas
VICT_DATASET = "datasets/vict/408v/data.csv"
_vict_df = None
def get_tri_from_dataset(victim_id):
    global _vict_df
    if _vict_df is None:
        _vict_df = pd.read_csv(VICT_DATASET)
        _vict_df = _vict_df.reset_index().rename(columns={"index": "victim_id"})
    row = _vict_df[_vict_df["victim_id"] == victim_id]
    if not row.empty:
        return row.iloc[0]["tri"]
    return None


def save_unified_map(unified, filepath=UNIFIED_FILE):
    """Salva o mapa unificado como TXT (compatível com visualizer)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "status", "floor_factor", "victim_present", "tri"])
        for cell in unified.values():
            writer.writerow([
                cell["x"],
                cell["y"],
                cell["status"],
                cell["floor_factor"],
                cell["victim_present"],
                cell["tri"],
            ])
    print(f"[UNIFY] Mapa unificado salvo em {filepath} ({len(unified)} células)")


def plot_unified_map(unified):
    """Exibe o mapa unificado com matplotlib (sem espelhamento vertical)."""
    if not unified:
        print("[UNIFY] Nada para plotar.")
        return

    xs = [c["x"] for c in unified.values()]
    ys = [c["y"] for c in unified.values()]

    width, height = max(xs) + 1, max(ys) + 1
    img = np.zeros((height, width, 3))

    for (x, y), cell in unified.items():
        if cell["status"] == "wall":
            color = (0, 0, 0)
        elif cell["status"] == "clear":
            color = (1, 1, 1)
        else:
            color = (0.7, 0.7, 0.7)

        if cell["victim_present"]:
            # 🟢 ADIÇÃO: colore conforme triagem
            tri = cell.get("tri", None)
            if tri == "green" or tri == 0:
                color = (0, 1, 0)
            elif tri == "yellow" or tri == 1:
                color = (1, 1, 0)
            elif tri == "red" or tri == 2:
                color = (1, 0, 0)
            elif tri == "black" or tri == 3:
                color = (0.2, 0.2, 0.2)
            else:
                color = (1, 0, 0)

        img[y, x] = color

    plt.figure(figsize=(8, 8))
    plt.imshow(img, origin="upper")
    plt.title("Mapa Unificado (com Triagem)")
    plt.axis("off")
    plt.show()


def unify_all_maps(outputs_dir=OUTPUT_DIR):
    """Lê todos os mapas individuais e gera o unificado"""
    csv_files = [
        os.path.join(outputs_dir, f)
        for f in os.listdir(outputs_dir)
        if f.startswith("map_explorer_") and f.endswith(".csv")
    ]

    if not csv_files:
        print("❌ Nenhum mapa encontrado em outputs/.")
        return

    print(f"[UNIFY] Lendo {len(csv_files)} mapas de exploradores...")
    maps = [read_agent_map(f) for f in csv_files]
    unified = unify_maps(maps)
    save_unified_map(unified, UNIFIED_FILE)
    plot_unified_map(unified)
