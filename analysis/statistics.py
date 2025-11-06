# analysis/statistics.py
import os
import glob
import pandas as pd

OUT_DIR = "outputs"

def victims_per_explorer():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "map_explorer_*.csv")))
    if not files:
        raise FileNotFoundError("Nenhum map_explorer_*.csv em outputs/")

    per_agent = {}
    all_detected = []
    for f in files:
        df = pd.read_csv(f)
        agent = os.path.splitext(os.path.basename(f))[0].replace("map_explorer_", "")
        v = df[(df.get("victim_present", 0) == 1)]
        v = v.dropna(subset=["victim_id"]) if "victim_id" in v.columns else v.head(0)
        if not v.empty and "victim_id" in v.columns:
            v["victim_id"] = v["victim_id"].astype(int)
            per_agent[agent] = v["victim_id"].nunique()
            all_detected.append(v[["victim_id"]])
        else:
            per_agent[agent] = 0

    if all_detected:
        ve = pd.concat(all_detected, ignore_index=True)["victim_id"].nunique()
    else:
        ve = 0

    return per_agent, ve

def overlap_metric():
    per_agent, ve = victims_per_explorer()
    total = sum(per_agent.values())
    if ve == 0:
        return per_agent, ve, None
    sobreposicao = (total / ve) - 1
    return per_agent, ve, sobreposicao

if __name__ == "__main__":
    per_agent, ve, s = overlap_metric()
    print("[STATS] Vítimas por explorador:", per_agent)
    print("[STATS] Vítimas únicas (Ve):", ve)
    print("[STATS] Sobreposição:", ("%.4f" % s) if s is not None else "N/A")
