"""Map representation utilities for explorer agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from vs.constants import VS


@dataclass
class MapCell:
    """Stores information about a single explored cell."""

    difficulty: float
    victim_seq: int
    obstacles: List[int]

    def merged_with(self, other: "MapCell") -> "MapCell":
        """Combine two map cells keeping the most informative data."""

        difficulty = self._merge_difficulty(self.difficulty, other.difficulty)
        victim_seq = self._merge_victim(self.victim_seq, other.victim_seq)
        obstacles = self._merge_obstacles(self.obstacles, other.obstacles)
        return MapCell(difficulty, victim_seq, obstacles)

    @staticmethod
    def _merge_difficulty(first: float, second: float) -> float:
        if first == VS.OBST_WALL:
            return second
        if second == VS.OBST_WALL:
            return first
        if first == 0:
            return second
        if second == 0:
            return first
        if first < 0:
            return second
        if second < 0:
            return first
        return min(first, second)

    @staticmethod
    def _merge_victim(first: int, second: int) -> int:
        if first != VS.NO_VICTIM:
            return first
        return second

    @staticmethod
    def _merge_obstacles(first: List[int], second: List[int]) -> List[int]:
        merged: List[int] = []
        for a, b in zip(first, second):
            if a == VS.UNK:
                merged.append(b)
            elif b == VS.UNK:
                merged.append(a)
            else:
                merged.append(a if a == b else b)
        return merged


class AgentMap:
    """Sparse representation of the explored portion of the grid."""

    def __init__(self) -> None:
        self._cells: Dict[Tuple[int, int], MapCell] = {}

    def keys(self) -> Iterable[Tuple[int, int]]:
        return self._cells.keys()

    def get(self, coord: Tuple[int, int]) -> MapCell | None:
        return self._cells.get(coord)

    def items(self) -> Iterable[Tuple[Tuple[int, int], MapCell]]:
        return self._cells.items()

    def update_cell(
        self,
        coord: Tuple[int, int],
        *,
        difficulty: float | None = None,
        victim_seq: int | None = None,
        obstacles: Iterable[int] | None = None,
    ) -> None:
        """Insert or update a cell in the map."""

        if coord not in self._cells:
            default_obstacles = [VS.UNK] * 8
            self._cells[coord] = MapCell(
                difficulty if difficulty is not None else 0.0,
                victim_seq if victim_seq is not None else VS.NO_VICTIM,
                list(obstacles) if obstacles is not None else default_obstacles,
            )
            return

        current = self._cells[coord]
        new_difficulty = current.difficulty if difficulty is None else difficulty
        new_victim = current.victim_seq if victim_seq is None else victim_seq
        if obstacles is None:
            new_obstacles = current.obstacles
        else:
            new_obstacles = MapCell._merge_obstacles(current.obstacles, list(obstacles))

        self._cells[coord] = MapCell(new_difficulty, new_victim, new_obstacles)

    def merge_from(self, other: "AgentMap") -> None:
        for coord, cell in other.items():
            if coord in self._cells:
                self._cells[coord] = self._cells[coord].merged_with(cell)
            else:
                self._cells[coord] = MapCell(cell.difficulty, cell.victim_seq, list(cell.obstacles))

    def as_sorted_rows(self) -> List[Tuple[int, int, MapCell]]:
        return sorted(((x, y, cell) for (x, y), cell in self._cells.items()), key=lambda item: (item[1], item[0]))
