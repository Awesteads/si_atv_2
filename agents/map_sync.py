"""Utilities for merging explorer maps and preparing data for rescuers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple


Position = Tuple[int, int]


@dataclass
class VictimInfo:
    """Information about a victim discovered by the explorers."""

    vid: int
    position: Position
    signals: List[float]
    sources: Set[str] = field(default_factory=set)

    def copy(self) -> "VictimInfo":
        return VictimInfo(
            vid=self.vid,
            position=tuple(self.position),
            signals=list(self.signals),
            sources=set(self.sources),
        )


@dataclass
class AgentLocalMap:
    """Snapshot of the area explored by one agent."""

    name: str
    base: Position
    visited: Set[Position]
    obstacles: Set[Position]
    victims: Dict[int, VictimInfo]

    def copy(self) -> "AgentLocalMap":
        return AgentLocalMap(
            name=self.name,
            base=self.base,
            visited=set(self.visited),
            obstacles=set(self.obstacles),
            victims={vid: info.copy() for vid, info in self.victims.items()},
        )


@dataclass
class UnifiedMap:
    """Combined map produced after the synchronization step."""

    base: Position
    visited: Set[Position] = field(default_factory=set)
    obstacles: Set[Position] = field(default_factory=set)
    victims: Dict[int, VictimInfo] = field(default_factory=dict)

    def copy(self) -> "UnifiedMap":
        return UnifiedMap(
            base=self.base,
            visited=set(self.visited),
            obstacles=set(self.obstacles),
            victims={vid: info.copy() for vid, info in self.victims.items()},
        )

    def as_dict(self) -> Dict:
        return {
            "base": list(self.base),
            "visited": [list(p) for p in sorted(self.visited)],
            "obstacles": [list(p) for p in sorted(self.obstacles)],
            "victims": [
                {
                    "id": vid,
                    "position": list(info.position),
                    "signals": list(info.signals),
                    "sources": sorted(info.sources),
                }
                for vid, info in sorted(self.victims.items())
            ],
        }


class MapSynchronizer:
    """Centralizes the logic for combining explorers' beliefs."""

    def __init__(self, grid_size: Tuple[int, int], base: Position):
        self._width, self._height = grid_size
        self._base = base

    # ------------------------------------------------------------------
    def merge(self, local_maps: Iterable[AgentLocalMap]) -> UnifiedMap:
        """Merge several local maps into a single shared map."""

        unified = UnifiedMap(base=self._base, visited={self._base})

        for local in local_maps:
            unified.visited.update(self._filter_positions(local.visited))
            unified.obstacles.update(self._filter_positions(local.obstacles))

            for vid, info in local.victims.items():
                if not self._within(info.position):
                    continue
                if vid not in unified.victims:
                    unified.victims[vid] = info.copy()
                else:
                    existing = unified.victims[vid]
                    existing.sources.update(info.sources)
                    if info.signals and not existing.signals:
                        existing.signals = list(info.signals)
                    existing.position = info.position

        unified.visited.difference_update(unified.obstacles)
        unified.visited.add(self._base)
        return unified

    # ------------------------------------------------------------------
    def save_to_json(self, unified_map: UnifiedMap, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(unified_map.as_dict(), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    def save_summary(self, unified_map: UnifiedMap, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.build_summary(unified_map))

    # ------------------------------------------------------------------
    def build_summary(self, unified_map: UnifiedMap) -> str:
        total_cells = self._width * self._height
        coverage = 0.0
        if total_cells:
            coverage = len(unified_map.visited) / total_cells

        victim_lines = []
        for vid, info in sorted(unified_map.victims.items()):
            src = ", ".join(sorted(info.sources)) if info.sources else "?"
            victim_lines.append(
                f"    Victim {vid:03d} at {info.position} (from: {src})"
            )

        lines = [
            "=== MAP SYNCHRONIZATION SUMMARY ===",
            f"Grid size: {self._width}x{self._height}",
            f"Base: {self._base}",
            f"Visited cells: {len(unified_map.visited)} ({coverage:.2%} coverage)",
            f"Known obstacles: {len(unified_map.obstacles)}",
            f"Victims located: {len(unified_map.victims)}",
        ]

        if victim_lines:
            lines.append("Victims detail:")
            lines.extend(victim_lines)
        else:
            lines.append("No victims located by surviving explorers.")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    def _within(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self._width and 0 <= y < self._height

    def _filter_positions(self, positions: Iterable[Position]) -> Set[Position]:
        return {p for p in positions if self._within(p)}


__all__ = ["VictimInfo", "AgentLocalMap", "UnifiedMap", "MapSynchronizer"]
