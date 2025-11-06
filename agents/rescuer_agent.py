"""Rescuer agent that operates using the synchronized map."""

from __future__ import annotations

import os
from collections import deque
from typing import Deque, Dict, Iterable, List, Set, Tuple

from vs.abstract_agent import AbstAgent
from vs.constants import VS

from agents.map_sync import UnifiedMap, VictimInfo


class RescuerAgent(AbstAgent):
    """Consumes the unified map to rescue known victims."""

    def __init__(self, env, config_file: str, shared_map: UnifiedMap):
        super().__init__(env, config_file)

        self._map = shared_map.copy()
        self._base = tuple(self.get_env().dic["BASE"])  # type: ignore[index]
        self._initialized = False

        self._pos = self._base
        self._action_queue: Deque[Dict] = deque()
        self._victim_order: List[int] = sorted(shared_map.victims.keys())
        self._completed_victims: Set[int] = set()
        self._unreachable_victims: Set[int] = set()
        self._unable_to_return = False

    # ------------------------------------------------------------------
    def deliberate(self) -> bool:
        if not self._initialized:
            self._initialize_plan()

        if self.get_rtime() < 0.0:
            self.set_state(VS.DEAD)
            return False

        while True:
            if not self._action_queue:
                if self._pos != self._base and not self._unable_to_return:
                    path_back = self._find_path(self._pos, self._base)
                    if path_back:
                        self._extend_with_path(path_back)
                        continue
                    self._unable_to_return = True

                if self._pos == self._base:
                    self.set_state(VS.ENDED)
                else:
                    self.set_state(VS.IDLE)
                return False

            action = self._action_queue.popleft()
            if action["type"] == "move":
                dx, dy, dest = action["dx"], action["dy"], action["dest"]
                res = self.walk(dx, dy)
                if res == VS.EXECUTED:
                    self._pos = dest
                    return True
                if res == VS.TIME_EXCEEDED:
                    self.set_state(VS.DEAD)
                    return False
                if res == VS.BUMPED:
                    self._register_obstacle(dest)
                    self._refresh_plan()
                    return True

            elif action["type"] == "aid":
                vid = action["victim"]
                res = self.first_aid()
                if res == VS.TIME_EXCEEDED:
                    self.set_state(VS.DEAD)
                    return False
                if res:
                    self._completed_victims.add(vid)
                else:
                    self._unreachable_victims.add(vid)
                return True

    # ------------------------------------------------------------------
    def _initialize_plan(self) -> None:
        self._initialized = True
        if not self._victim_order:
            return
        victims = [self._map.victims[vid] for vid in self._victim_order]
        plan = self._plan_sequence(self._pos, victims)
        self._action_queue.extend(plan)

    # ------------------------------------------------------------------
    def _plan_sequence(self, start: Tuple[int, int], victims: Iterable[VictimInfo]):
        plan: Deque[Dict] = deque()
        current = start
        for info in victims:
            if info.vid in self._completed_victims or info.vid in self._unreachable_victims:
                continue
            if info.position not in self._map.visited:
                self._unreachable_victims.add(info.vid)
                continue
            path = self._find_path(current, info.position)
            if not path:
                self._unreachable_victims.add(info.vid)
                continue
            plan.extend(self._path_to_actions(path))
            plan.append({"type": "aid", "victim": info.vid})
            current = info.position

        if current != self._base:
            path_back = self._find_path(current, self._base)
            if path_back:
                plan.extend(self._path_to_actions(path_back))
            else:
                self._unable_to_return = True
        return plan

    # ------------------------------------------------------------------
    def _find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]

        allowed = self._map.visited - self._map.obstacles
        if start not in allowed or goal not in allowed:
            return []

        queue: Deque[Tuple[int, int]] = deque([start])
        parents: Dict[Tuple[int, int], Tuple[int, int] | None] = {start: None}

        while queue:
            current = queue.popleft()
            for neigh in self._neighbors(current):
                if neigh in parents:
                    continue
                if neigh not in allowed:
                    continue
                parents[neigh] = current
                if neigh == goal:
                    queue.clear()
                    break
                queue.append(neigh)

        if goal not in parents:
            return []

        path = []
        node: Tuple[int, int] | None = goal
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    def _neighbors(self, pos: Tuple[int, int]):
        x, y = pos
        for dx, dy in AbstAgent.AC_INCR.values():
            yield (x + dx, y + dy)

    # ------------------------------------------------------------------
    def _path_to_actions(self, path: List[Tuple[int, int]]):
        actions: Deque[Dict] = deque()
        for i in range(1, len(path)):
            prev = path[i - 1]
            nxt = path[i]
            dx = nxt[0] - prev[0]
            dy = nxt[1] - prev[1]
            actions.append({"type": "move", "dx": dx, "dy": dy, "dest": nxt})
        return actions

    # ------------------------------------------------------------------
    def _extend_with_path(self, path: List[Tuple[int, int]]) -> None:
        self._action_queue.extend(self._path_to_actions(path))

    # ------------------------------------------------------------------
    def _register_obstacle(self, pos: Tuple[int, int]) -> None:
        self._map.obstacles.add(pos)
        if pos in self._map.visited:
            self._map.visited.remove(pos)

    # ------------------------------------------------------------------
    def _refresh_plan(self) -> None:
        remaining = [
            self._map.victims[vid]
            for vid in self._victim_order
            if vid not in self._completed_victims
        ]
        self._action_queue.clear()
        self._action_queue.extend(self._plan_sequence(self._pos, remaining))

    # ------------------------------------------------------------------
    def save_results(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Rescuer: {self.NAME}\n")
            f.write(f"Completed victims: {sorted(self._completed_victims)}\n")
            f.write(f"Unreachable victims: {sorted(self._unreachable_victims)}\n")
            f.write(f"Returned to base: {self._pos == self._base and not self._unable_to_return}\n")


__all__ = ["RescuerAgent"]
