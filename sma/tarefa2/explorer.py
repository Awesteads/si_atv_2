"""Explorer agents implementing an online DFS strategy."""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Sequence, Tuple

from vs.abstract_agent import AbstAgent
from vs.constants import VS

from .map_representation import AgentMap
from .ml_models import VictimModels


Direction = int


class ExplorerDFS(AbstAgent):
    """Explorer that performs an online depth-first search of the environment."""

    SAFETY_MARGIN = 8.0

    def __init__(
        self,
        env,
        config_file: str,
        coordinator,
        models: VictimModels,
        direction_preference: Sequence[Direction] | None = None,
    ) -> None:
        super().__init__(env, config_file)
        self.coordinator = coordinator
        self.models = models

        self.map = AgentMap()
        self.rel_x = 0
        self.rel_y = 0
        self.walk_history: List[Tuple[int, int, float]] = []
        self.accumulated_return_cost = 0.0
        self.steps_taken = 0
        self.path_stack: List[Tuple[int, int]] = [(0, 0)]
        self.untried: Dict[Tuple[int, int], Deque[Direction]] = {}
        self.found_victims: Dict[int, Dict[str, object]] = {}
        self.returning = False
        self.finished = False
        self.direction_preference = list(direction_preference) if direction_preference else list(range(8))
        self.direction_order = {direction: idx for idx, direction in enumerate(self.direction_preference)}

        self.set_state(VS.ACTIVE)
        self._update_current_cell(initial=True)
        if hasattr(self.coordinator, "register_explorer"):
            self.coordinator.register_explorer(self.NAME)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _ordered_clear_directions(self) -> Deque[Direction]:
        obstacles = self.check_walls_and_lim()
        clear_dirs = [idx for idx, result in enumerate(obstacles) if result == VS.CLEAR]
        ordered = sorted(
            clear_dirs,
            key=lambda direction: self.direction_order.get(direction, len(self.direction_preference) + direction),
        )
        return deque(ordered)

    def _update_current_cell(self, initial: bool = False) -> None:
        obstacles = self.check_walls_and_lim()
        victim_seq = self.check_for_victim()
        existing = self.map.get((self.rel_x, self.rel_y))
        difficulty = existing.difficulty if existing else 0.0
        if not initial and self.walk_history:
            difficulty = self._current_cell_difficulty()
        self.map.update_cell((self.rel_x, self.rel_y), difficulty=difficulty, victim_seq=victim_seq, obstacles=obstacles)
        if victim_seq != VS.NO_VICTIM and victim_seq not in self.found_victims:
            self._handle_victim(victim_seq)

    def _current_cell_difficulty(self) -> float:
        if not self.walk_history:
            return 0.0
        last_dx, last_dy, last_cost = self.walk_history[-1]
        if last_dx == 0 and last_dy == 0:
            return 0.0
        return last_cost / (self.COST_DIAG if last_dx and last_dy else self.COST_LINE)

    def _handle_victim(self, seq: int) -> None:
        vital_signals = self.read_vital_signals()
        if vital_signals == VS.TIME_EXCEEDED:
            self.returning = True
            return
        if not vital_signals:
            return
        tri_prediction, sobr_prediction = self.models.predict(vital_signals)
        features = {
            "idade": vital_signals[1],
            "fc": vital_signals[2],
            "fr": vital_signals[3],
            "pas": vital_signals[4],
            "spo2": vital_signals[5],
            "temp": vital_signals[6],
            "pr": vital_signals[7],
            "sg": vital_signals[8],
            "fx": vital_signals[9],
            "queim": vital_signals[10],
            "gcs": vital_signals[11],
            "avpu": vital_signals[12],
        }
        self.found_victims[seq] = {
            "id": seq,
            "position": (self.rel_x, self.rel_y),
            "tri": tri_prediction,
            "sobr": sobr_prediction,
            "features": features,
        }

    def _next_direction(self, state: Tuple[int, int]) -> Direction | None:
        if state not in self.untried:
            self.untried[state] = self._ordered_clear_directions()
        if self.untried[state]:
            return self.untried[state].popleft()
        return None

    def _push_walk(self, dx: int, dy: int, cost: float) -> None:
        self.walk_history.append((dx, dy, cost))
        self.accumulated_return_cost += cost
        self.path_stack.append((self.rel_x, self.rel_y))

    def _pop_walk(self) -> Tuple[int, int, float] | None:
        if not self.walk_history:
            return None
        dx, dy, cost = self.walk_history.pop()
        self.accumulated_return_cost = max(0.0, self.accumulated_return_cost - cost)
        if self.path_stack:
            self.path_stack.pop()
        return dx, dy, cost

    def _should_start_return(self) -> bool:
        remaining = self.get_rtime()
        return_cost = self.accumulated_return_cost
        return remaining <= return_cost + self.SAFETY_MARGIN

    def _walk(self, dx: int, dy: int) -> int:
        before = self.get_rtime()
        result = self.walk(dx, dy)
        if result == VS.EXECUTED:
            after = self.get_rtime()
            step_cost = before - after
            self.rel_x += dx
            self.rel_y += dy
            self._push_walk(dx, dy, step_cost)
            self.steps_taken += 1
            self._update_current_cell()
        elif result == VS.BUMPED:
            self._update_current_cell()
        elif result == VS.TIME_EXCEEDED:
            self.returning = True
        return result

    def _backtrack_one_step(self) -> bool:
        last = self._pop_walk()
        if last is None:
            return False
        dx, dy, _ = last
        result = super().walk(-dx, -dy)
        if result == VS.EXECUTED:
            self.rel_x -= dx
            self.rel_y -= dy
            self.steps_taken += 1
            self._update_current_cell()
            return True
        return False

    def _return_to_base(self) -> bool:
        if not self.walk_history:
            if not self.finished:
                self.finished = True
                self.coordinator.receive_report(
                    self.NAME,
                    self.map,
                    list(self.found_victims.values()),
                    {
                        "steps": self.steps_taken,
                        "victim_ids": list(self.found_victims.keys()),
                    },
                )
            return False
        last = self._pop_walk()
        if last is None:
            return False
        dx, dy, _ = last
        result = super().walk(-dx, -dy)
        if result == VS.EXECUTED:
            self.rel_x -= dx
            self.rel_y -= dy
            self.steps_taken += 1
            self._update_current_cell()
            return True
        return False

    # ------------------------------------------------------------------
    # Deliberation loop
    # ------------------------------------------------------------------
    def deliberate(self) -> bool:
        if self.returning or self._should_start_return():
            self.returning = True
            return self._return_to_base()

        current_state = (self.rel_x, self.rel_y)
        direction = self._next_direction(current_state)

        if direction is not None:
            dx, dy = AbstAgent.AC_INCR[direction]
            result = self._walk(dx, dy)
            if result == VS.EXECUTED:
                return True
            if result == VS.TIME_EXCEEDED:
                return self._return_to_base()
            return True

        if len(self.path_stack) <= 1:
            self.returning = True
            return self._return_to_base()

        if self._backtrack_one_step():
            return True

        self.returning = True
        return self._return_to_base()
