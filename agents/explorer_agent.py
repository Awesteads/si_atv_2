import os
from typing import Dict

from vs.abstract_agent import AbstAgent
from vs.constants import VS

from agents.map_sync import AgentLocalMap, VictimInfo


class ExplorerAgent(AbstAgent):
    """
    Explorer que executa uma DFS online (um passo por ciclo),
    registrando vítimas/obstáculos e preservando energia para
    retornar à base antes do TLIM.
    """

    def __init__(self, env, config_file, seed=None):
        super().__init__(env, config_file)

        # Estado interno do algoritmo (incremental: 1 ação por deliberate)
        self._rng = (
            __import__("random").Random(seed)
            if seed is not None
            else __import__("random")
        )
        self._initialized = False

        # Rastros de exploração
        self.victims_found = set()  # {(x, y)}
        self.obstacles_found = set()  # {(x, y)}
        self.visited = set()  # {(x, y)}
        self.victim_data: Dict[int, VictimInfo] = {}

        # Pilha do DFS e posição atual (rastreamos localmente)
        self.stack = []
        self.pos = None  # (x, y)

        # Controle de retorno/base
        self.returning = False
        self.return_path = []  # caminho (lista de (x,y)) para retornar (greedy + backtrack)
        self._base = tuple(self.get_env().dic["BASE"])  # (x_base, y_base)

        # Mapeamento de movimentos (mesma convenção de AbstAgent.AC_INCR)
        # 0:u, 1:ur, 2:r, 3:dr, 4:d, 5:dl, 6:l, 7:ul
        self._dirs = list(AbstAgent.AC_INCR.items())

    # ---------- helpers ----------

    def _step_cost_lower_bound(self, p1, p2):
        """Aproximação de custo para estimar retorno (linha vs diag)."""
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        # Heurística: no pior caso, considera todos os passos como 'line'.
        # (preserva mais energia do que o estritamente necessário)
        return (dx + dy) * self.COST_LINE

    def _neighbors_clear(self, pos):
        """Usa sensores locais para listar vizinhos livres e marcar obstáculos/limites."""
        x, y = pos
        obst = self.check_walls_and_lim()  # vetor 8 direções (CLEAR/WALL/END)
        env = self.get_env()

        free = []
        for i, (dx, dy) in self._dirs:
            tx, ty = x + dx, y + dy
            # Registra obstáculos/limites (quando vistos)
            if 0 <= tx < env.dic["GRID_WIDTH"] and 0 <= ty < env.dic["GRID_HEIGHT"]:
                if obst[i] == VS.WALL:
                    self.obstacles_found.add((tx, ty))
            # Vizinhos transitáveis
            if obst[i] == VS.CLEAR:
                free.append((tx, ty))
        return free

    def _choose_unvisited(self, neighbors):
        unvisited = [n for n in neighbors if n not in self.visited]
        if not unvisited:
            return None
        return self._rng.choice(unvisited)

    def _greedy_step_towards(self, src, dst):
        """Um passo ganancioso na direção do destino (pode bater)."""
        x, y = src
        bx, by = dst
        dx = 0 if bx == x else (1 if bx > x else -1)
        dy = 0 if by == y else (1 if by > y else -1)
        return (x + dx, y + dy), (dx, dy)

    # ---------- ciclo de decisão (1 ação por chamada) ----------

    def deliberate(self) -> bool:
        """
        Escolhe e executa UMA ação de caminhada por ciclo, retornando:
          True  -> ainda há ações a realizar
          False -> terminou (sem ações pendentes)
        """
        # Inicialização lazy no primeiro ciclo
        if not self._initialized:
            self.pos = self._base
            self.stack = [self.pos]
            self.visited = set()
            self.returning = False
            self.return_path = []
            self._initialized = True

        # Se acabou o tempo, não há mais o que fazer
        if self.get_rtime() < 0.0:
            self.set_state(VS.DEAD)
            return False

        # Marca visita / detecta vítima
        self.visited.add(self.pos)
        vic_id = self.check_for_victim()
        if vic_id != VS.NO_VICTIM:
            vitals = self.read_vital_signals()  # CONSOME TEMPO (TEMOS QUE REPLANEJAR)
            if vitals == VS.TIME_EXCEEDED:
                self.set_state(VS.DEAD)
                return False
            if vitals:
                self.victims_found.add((self.pos[0], self.pos[1], vic_id))
                info = VictimInfo(
                    vid=vic_id,
                    position=self.pos,
                    signals=list(vitals),
                    sources={self.NAME},
                )
                self.victim_data[vic_id] = info

        # Se estamos em modo de retorno, tenta dar 1 passo para base
        # MELHORAR ISSO PARECE QUE ESTÁ FAZENDO MAIS DE 1 AÇÃO POR CICLO
        if self.returning:
            # 1) tentativa gananciosa
            target = self._base
            nxt_pos, (dx, dy) = self._greedy_step_towards(self.pos, target)
            res = self.walk(dx, dy)
            if res == VS.EXECUTED:
                self.pos = nxt_pos
                # terminou retorno?
                if self.pos == self._base:
                    self.set_state(VS.ENDED)
                    return False
                return True  # continuará retornando em próximos ciclos
            else:
                # bateu: registra obstáculo observado naquela casa
                self.obstacles_found.add(nxt_pos)
                # 2) fallback: backtrack via pilha se houver
                if len(self.stack) > 1:
                    # volta 1 passo na pilha
                    prev = self.stack[-2]
                    dx = prev[0] - self.pos[0]
                    dy = prev[1] - self.pos[1]
                    res = self.walk(dx, dy)
                    if res == VS.EXECUTED:
                        self.pos = prev
                        self.stack.pop()
                        return True
                # Sem como mover: encerra
                self.set_state(VS.IDLE)
                return False

        # Não retornando: decide próxima expansão DFS
        neighbors = self._neighbors_clear(self.pos)

        # Reserva de energia simples: se o custo “mínimo” p/ voltar já ameaça TLIM, inicie retorno
        if self._step_cost_lower_bound(self.pos, self._base) >= self.get_rtime():
            self.returning = True
            return True  # próximo ciclo tentará caminhar de volta

        # Escolhe vizinho não visitado
        nxt = self._choose_unvisited(neighbors)

        if nxt is not None:
            # Checa novamente reserva de energia antes de mover
            if self._step_cost_lower_bound(nxt, self._base) >= self.get_rtime():
                self.returning = True
                return True

            # Executa 1 passo
            dx = nxt[0] - self.pos[0]
            dy = nxt[1] - self.pos[1]
            res = self.walk(dx, dy)
            if res == VS.EXECUTED:
                self.pos = nxt
                self.stack.append(nxt)
                return True
            else:
                # Bateu: marca obstáculo
                self.obstacles_found.add(nxt)
                # fica no mesmo lugar neste ciclo
                return True

        # Nenhum vizinho novo: backtrack
        if len(self.stack) > 1:
            prev = self.stack[-2]
            # Reserva antes de voltar (em tese voltar aproxima da base, então seguro)
            dx = prev[0] - self.pos[0]
            dy = prev[1] - self.pos[1]
            res = self.walk(dx, dy)
            if res == VS.EXECUTED:
                self.pos = prev
                self.stack.pop()
                return True
            else:
                # algo impediu voltar pela aresta (parede dinâmica? penaliza e tenta retorno à base)
                self.returning = True
                return True

        # Pilha acabou: iniciar retorno final
        if self.pos != self._base:
            self.returning = True
            return True

        # Já na base e nada a explorar
        self.set_state(VS.ENDED)
        return False

    # ---------- utilidades compatíveis com sua versão antiga ----------

    def get_local_map(self) -> AgentLocalMap:
        """Return a snapshot of the explored region."""

        return AgentLocalMap(
            name=self.NAME,
            base=self._base,
            visited=set(self.visited),
            obstacles=set(self.obstacles_found),
            victims={vid: info.copy() for vid, info in self.victim_data.items()},
        )

    def save_results(self, victims_path):
        """
        Exporta resultados como no seu Explorer antigo:
        - vítimas: (x,y,9)
        - obstáculos: (x,y,-1) em arquivo *_obst.txt
        """
        import csv

        obst_path = victims_path.replace(".txt", "_obst.txt")
        os.makedirs(os.path.dirname(victims_path), exist_ok=True)
        with open(victims_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["victim_id", "x", "y", "signals..."])
            for vid, info in sorted(self.victim_data.items()):
                writer.writerow([vid, info.position[0], info.position[1], *info.signals])
        with open(obst_path, "w", newline="") as f:
            w = csv.writer(f)
            for x, y in sorted(self.obstacles_found):
                w.writerow([x, y, -1])
