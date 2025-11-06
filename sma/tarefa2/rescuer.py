"""Rescuer agents responsible for clustering and assignments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean
from typing import Dict, List, Sequence, Tuple

from vs.abstract_agent import AbstAgent
from vs.constants import VS

from .map_representation import AgentMap


@dataclass
class ClusterAssignment:
    cluster_id: int
    victims: List[Dict[str, object]]


class RescuerBase(AbstAgent):
    def __init__(self, env, config_file: str, coordinator, *, is_master: bool = False) -> None:
        super().__init__(env, config_file)
        self.coordinator = coordinator
        self.is_master = is_master
        self.pending_assignment: ClusterAssignment | None = None
        self.set_state(VS.IDLE)
        if hasattr(self.coordinator, "register_rescuer"):
            self.coordinator.register_rescuer(self)

    def deliberate(self) -> bool:  # pragma: no cover - base class
        if self.pending_assignment is None:
            return False
        return False

    def receive_assignment(self, assignment: ClusterAssignment) -> None:
        self.pending_assignment = assignment
        self.set_state(VS.ACTIVE)


class RescuerWorker(RescuerBase):
    def deliberate(self) -> bool:
        if self.pending_assignment is None:
            self.set_state(VS.ENDED)
            return False

        cluster_id = self.pending_assignment.cluster_id
        victims = self.pending_assignment.victims
        print(f"{self.NAME}: designado para cluster {cluster_id} contendo {len(victims)} vítimas")
        for victim in victims:
            pos = victim["position"]
            tri = victim["tri"]
            sobr = victim["sobr"]
            print(f"  vítima {victim['id']} em {pos} tri={tri} sobr={sobr:.2f}")
        self.pending_assignment = None
        return False


class RescuerMaster(RescuerBase):
    def __init__(self, env, config_file: str, coordinator, output_dir: Path, nb_clusters: int = 3) -> None:
        super().__init__(env, config_file, coordinator, is_master=True)
        self.output_dir = Path(output_dir)
        self.nb_clusters = nb_clusters
        self.workers: List[RescuerBase] = []
        self.global_map: AgentMap | None = None
        self.victims: List[Dict[str, object]] = []
        self.metrics: Dict[str, object] = {}
        self.model_metrics: Dict[str, float] = {}
        self._ready = False

    def prepare_mission(
        self,
        global_map: AgentMap,
        victims: List[Dict[str, object]],
        workers: List[RescuerBase],
        metrics: Dict[str, object],
        model_metrics: Dict[str, float],
    ) -> None:
        self.global_map = global_map
        self.victims = victims
        self.workers = workers
        self.metrics = metrics
        self.model_metrics = model_metrics
        self._ready = True
        self.set_state(VS.ACTIVE)

    def deliberate(self) -> bool:
        if not self._ready:
            return False

        clusters = self._cluster_victims(self.victims, self.nb_clusters)
        self._write_cluster_files(clusters)
        assignments = self._assign_clusters(clusters)
        self._print_summary(assignments)
        self._ready = False
        return False

    # ------------------------------------------------------------------
    # Clustering helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_features(features: Sequence[Sequence[float]]) -> List[List[float]]:
        mins = [min(column) for column in zip(*features)]
        maxs = [max(column) for column in zip(*features)]
        normalized: List[List[float]] = []
        for row in features:
            normalized.append([
                0.0 if mn == mx else (value - mn) / (mx - mn)
                for value, mn, mx in zip(row, mins, maxs)
            ])
        return normalized

    def _cluster_victims(self, victims: List[Dict[str, object]], nb_clusters: int) -> Dict[int, List[Dict[str, object]]]:
        if not victims:
            return {i: [] for i in range(nb_clusters)}

        rng = Random(42)
        features = [[victim["position"][0], victim["position"][1], float(victim["tri"]), float(victim["sobr"])] for victim in victims]
        normalized = self._normalize_features(features)
        indices = list(range(len(victims)))
        rng.shuffle(indices)
        centroids = [list(normalized[i]) for i in indices[:nb_clusters]]
        if len(centroids) < nb_clusters:
            # duplicate last centroid if there are fewer victims than clusters
            while len(centroids) < nb_clusters:
                centroids.append(list(centroids[-1]))

        assignments = [0] * len(victims)
        for _ in range(100):
            changed = False
            # assignment step
            for idx, point in enumerate(normalized):
                distances = [sum((p - c) ** 2 for p, c in zip(point, centroid)) for centroid in centroids]
                best_cluster = distances.index(min(distances))
                if assignments[idx] != best_cluster:
                    assignments[idx] = best_cluster
                    changed = True
            # update step
            new_centroids: List[List[float]] = []
            for cluster_idx in range(nb_clusters):
                cluster_points = [normalized[i] for i, c in enumerate(assignments) if c == cluster_idx]
                if cluster_points:
                    centroid = [mean(values) for values in zip(*cluster_points)]
                else:
                    centroid = list(centroids[cluster_idx])
                new_centroids.append(centroid)
            if not changed:
                break
            centroids = new_centroids

        clusters: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(nb_clusters)}
        for assignment, victim in zip(assignments, victims):
            clusters[assignment].append(victim)
        return clusters

    def _write_cluster_files(self, clusters: Dict[int, List[Dict[str, object]]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(self.nb_clusters):
            victims = clusters.get(idx, [])
            path = self.output_dir / f"cluster{idx + 1}.txt"
            with path.open("w", encoding="utf-8") as file:
                file.write("id_vict,x,y,sobr,tri\n")
                for victim in victims:
                    x, y = victim["position"]
                    file.write(f"{victim['id']},{x},{y},{victim['sobr']:.2f},{victim['tri']}\n")

    def _assign_clusters(self, clusters: Dict[int, List[Dict[str, object]]]) -> List[Tuple[RescuerBase, ClusterAssignment]]:
        rescuers: List[RescuerBase] = [self] + self.workers
        assignments: List[Tuple[RescuerBase, ClusterAssignment]] = []
        for idx in range(self.nb_clusters):
            victims = clusters.get(idx, [])
            rescuer = rescuers[idx % len(rescuers)]
            assignment = ClusterAssignment(cluster_id=idx + 1, victims=victims)
            if rescuer is self:
                self.pending_assignment = assignment
            else:
                rescuer.receive_assignment(assignment)
            assignments.append((rescuer, assignment))
        return assignments

    def _print_summary(self, assignments: List[Tuple[RescuerBase, ClusterAssignment]]) -> None:
        print(f"{self.NAME}: resumo dos agrupamentos e atribuições")
        if self.metrics:
            print("Resultados da exploração:")
            for key, value in self.metrics.items():
                print(f"  {key}: {value}")
        if self.model_metrics:
            print("Desempenho dos modelos de ML (validação hold-out):")
            print(f"  Acurácia triagem: {self.model_metrics['tri_accuracy']:.3f}")
            print(f"  MAE sobrevivência: {self.model_metrics['sobr_mae']:.3f}")
        for rescuer, assignment in assignments:
            label = "mestre" if rescuer is self else "apoio"
            print(
                f"  {rescuer.NAME} ({label}) receberá cluster {assignment.cluster_id} com {len(assignment.victims)} vítimas"
            )
        self.pending_assignment = None
        for worker in self.workers:
            if worker.pending_assignment is None:
                worker.set_state(VS.ENDED)
