"""Mission coordinator responsible for merging maps and orchestrating agents."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .map_representation import AgentMap


class MissionCoordinator:
    def __init__(
        self,
        *,
        total_explorers: int,
        nb_clusters: int,
        output_dir: Path,
        model_metrics: Dict[str, float],
    ) -> None:
        self.total_explorers = total_explorers
        self.nb_clusters = nb_clusters
        self.output_dir = Path(output_dir)
        self.model_metrics = model_metrics
        self.explorer_reports: List[tuple[str, AgentMap, List[Dict[str, object]], Dict[str, object]]] = []
        self.explorer_names: List[str] = []
        self.rescuers: List[object] = []
        self.master = None
        self.global_map = AgentMap()
        self.metrics: Dict[str, object] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_explorer(self, name: str) -> None:
        if name not in self.explorer_names:
            self.explorer_names.append(name)

    def register_rescuer(self, rescuer) -> None:
        if getattr(rescuer, "is_master", False):
            self.master = rescuer
        else:
            self.rescuers.append(rescuer)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def receive_report(
        self,
        explorer_name: str,
        agent_map: AgentMap,
        victims: List[Dict[str, object]],
        stats: Dict[str, object],
    ) -> None:
        self.explorer_reports.append((explorer_name, agent_map, victims, stats))
        if len(self.explorer_reports) == self.total_explorers:
            self._finalize_reports()

    def _finalize_reports(self) -> None:
        aggregated_victims: Dict[int, Dict[str, object]] = {}
        victim_sets: Dict[str, set[int]] = {}
        for name, agent_map, victims, stats in self.explorer_reports:
            self.global_map.merge_from(agent_map)
            ids = set(int(victim_id) for victim_id in stats.get("victim_ids", []))
            victim_sets[name] = ids
            for victim in victims:
                vid = int(victim["id"])
                current = aggregated_victims.get(vid)
                if current is None or victim["tri"] > current["tri"]:
                    aggregated_victims[vid] = victim

        unique_victims = set().union(*victim_sets.values()) if victim_sets else set()
        total_found = sum(len(ids) for ids in victim_sets.values())
        overlap = (total_found / len(unique_victims) - 1.0) if unique_victims else 0.0
        per_explorer = {name: len(ids) for name, ids in victim_sets.items()}

        self.metrics = {
            "vítimas únicas": len(unique_victims),
            "vítimas por explorador": per_explorer,
            "sobreposição": round(overlap, 4),
        }

        if self.master is not None:
            self.master.prepare_mission(
                self.global_map,
                list(aggregated_victims.values()),
                self.rescuers,
                self.metrics,
                self.model_metrics,
            )
