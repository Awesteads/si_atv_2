# analysis/merge_maps.py
import os
import csv
from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs"
UNIFIED_FILE = os.path.join(OUTPUT_DIR, "map_unificado.csv")

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

        unified[xy] = {
            "x": xy[0],
            "y": xy[1],
            "status": chosen_status,
            "floor_factor": avg_floor,
            "victim_present": int(victim_present),
        }

    return unified


def save_unified_map(unified, filepath=UNIFIED_FILE):
    """Salva o mapa unificado como CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "status", "floor_factor", "victim_present"])
        for cell in unified.values():
            writer.writerow([
                cell["x"],
                cell["y"],
                cell["status"],
                cell["floor_factor"],
                cell["victim_present"],
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
            color = (1, 0, 0)  # vermelho

        # não inverter manualmente — vamos deixar o imshow corrigir
        img[y, x] = color

    plt.figure(figsize=(8, 8))
    plt.imshow(img, origin='upper')  # 🔹 ESSENCIAL: pygame-style
    plt.title("Mapa Unificado (Exploradores)")
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
