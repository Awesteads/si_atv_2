# analysis/merge_maps.py
import os
import csv
import json
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from agents.map_structures import CellInfo, VitalSigns, NeighborSet, read_map_csv

OUTPUT_DIR = "outputs"
UNIFIED_FILE = os.path.join(OUTPUT_DIR, "map_unificado.txt")

# prioridade de status (para resolver conflitos)
STATUS_PRIORITY = {"wall": 3, "clear": 2, "unknown": 1, "out_of_bounds": 0}


# ============================================================
# Triagem a partir de sinais vitais
# ============================================================
def infer_triage_color(vitals):
    """
    Infere cor de triagem (green/yellow/red/black) a partir dos sinais vitais.
    Aceita objeto VitalSigns, lista ou string JSON.
    """
    if not vitals:
        return None

    # converte para lista se for VitalSigns
    if hasattr(vitals, "as_list"):
        vitals = vitals.as_list()

    # converte de JSON se for string
    if isinstance(vitals, str):
        try:
            vitals = json.loads(vitals)
        except Exception:
            return None

    if not isinstance(vitals, (list, tuple)) or not vitals:
        return None

    try:
        temp = vitals[6] if len(vitals) > 6 else 37
        spo2 = vitals[1] if len(vitals) > 1 else 98
        hr = vitals[0] if len(vitals) > 0 else 80

        if temp < 34 or spo2 < 85 or hr < 40:
            return "black"
        elif temp < 36 or spo2 < 90 or hr < 50:
            return "red"
        elif temp < 37 or spo2 < 95 or hr < 60:
            return "yellow"
        else:
            return "green"
    except Exception:
        return None


# ============================================================
# Unificação
# ============================================================
def unify_maps(maps):
    """Une múltiplos mapas de agentes com base em CellInfo."""
    unified = {}
    all_coords = set().union(*(grid.keys() for grid in maps))

    for xy in all_coords:
        cells = [g[xy] for g in maps if xy in g]

        # status com maior prioridade
        chosen_status = max(
            (c.status for c in cells),
            key=lambda s: STATUS_PRIORITY.get(s, 0),
        )

        # médias dos campos numéricos
        floors = [c.floor_factor for c in cells if c.floor_factor is not None]
        avg_floor = mean(floors) if floors else None

        g_cost_vals = [c.g_cost for c in cells if c.g_cost is not None]
        avg_g_cost = mean(g_cost_vals) if g_cost_vals else None

        last_seen_steps = [c.last_seen_step for c in cells if c.last_seen_step >= 0]
        last_seen_avg = mean(last_seen_steps) if last_seen_steps else None

        # campos booleanos e de presença
        victim_present = any(c.victim_present for c in cells)
        victim_id = next((c.victim_id for c in cells if c.victim_id), None)
        vitals_read = any(c.vitals_read for c in cells)

        # unificação de vitals_raw (objeto VitalSigns)
        vitals_raw_obj = next((c.vitals_raw for c in cells if c.vitals_raw and len(c.vitals_raw) > 0), None)

        # vizinhos (unir conjuntos de NeighborSet)
        all_neighbors = NeighborSet()
        for c in cells:
            if c.neighbors_clear:
                all_neighbors.update(c.neighbors_clear)

        # inferência de triagem
        tri_color = infer_triage_color(vitals_raw_obj)

        unified[xy] = {
            "x": xy[0],
            "y": xy[1],
            "status": chosen_status,
            "visited": int(any(c.visited for c in cells)),
            "last_seen_step": last_seen_avg,
            "floor_factor": avg_floor,
            "victim_present": int(victim_present),
            "victim_id": victim_id,
            "vitals_read": int(vitals_read),
            "vitals_raw": vitals_raw_obj.to_json() if vitals_raw_obj else "[]",
            "g_cost": avg_g_cost,
            "parent_x": None,
            "parent_y": None,
            "neighbors_clear": all_neighbors.to_json(),
            "tri_color": tri_color,
        }

    return unified


# ============================================================
# Salvamento e visualização
# ============================================================
def save_unified_map(unified, filepath=UNIFIED_FILE):
    """Salva o mapa unificado com todos os campos, incluindo vitals_raw."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "x", "y", "status", "visited", "last_seen_step", "floor_factor",
            "victim_present", "victim_id", "vitals_read", "vitals_raw",
            "g_cost", "parent_x", "parent_y", "neighbors_clear", "tri_color"
        ])
        for cell in unified.values():
            writer.writerow([
                cell["x"], cell["y"], cell["status"], cell["visited"], cell["last_seen_step"],
                cell["floor_factor"], cell["victim_present"], cell["victim_id"],
                cell["vitals_read"], cell["vitals_raw"], cell["g_cost"],
                cell["parent_x"], cell["parent_y"], cell["neighbors_clear"], cell["tri_color"]
            ])
    print(f"[UNIFY] Mapa unificado salvo em {filepath} ({len(unified)} células)")


def plot_unified_map(unified):
    """Exibe o mapa unificado colorido, com triagem baseada em vitals_raw."""
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

        if int(cell["victim_present"]):
            tri = cell.get("tri_color", None)
            if tri == "green":
                color = (0, 1, 0)
            elif tri == "yellow":
                color = (1, 1, 0)
            elif tri == "red":
                color = (1, 0, 0)
            elif tri == "black":
                color = (0.2, 0.2, 0.2)

        img[y, x] = color

    plt.figure(figsize=(8, 8))
    plt.imshow(img, origin="upper")
    plt.title("Mapa Unificado (Triagem por vitals_raw)")
    plt.axis("off")
    plt.show()


def unify_all_maps(outputs_dir=OUTPUT_DIR):
    """Lê todos os mapas CSV e gera o mapa unificado colorido."""
    csv_files = [
        os.path.join(outputs_dir, f)
        for f in os.listdir(outputs_dir)
        if f.startswith("map_explorer_") and f.endswith(".csv")
    ]
    if not csv_files:
        print("❌ Nenhum mapa encontrado em outputs/.")
        return

    print(f"[UNIFY] Lendo {len(csv_files)} mapas exportados...")
    maps = [read_map_csv(f) for f in csv_files]  # agora lê CellInfo diretamente
    unified = unify_maps(maps)
    save_unified_map(unified, UNIFIED_FILE)
    plot_unified_map(unified)
