# agents/map_structures.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Iterable
import os
import csv
import json

# Tipos
Status = str
Coord = Tuple[int, int]
MapGrid = Dict[Coord, "CellInfo"]

# ================================================================
# Classe auxiliar para vizinhos (encapsula o Set[Coord])
# ================================================================
class NeighborSet:
    """Encapsula conjunto de coordenadas e oferece serialização legível."""

    def __init__(self, coords: Optional[Iterable[Coord]] = None):
        self._coords = set(coords) if coords else set()

    def add(self, xy: Coord) -> None:
        self._coords.add(xy)

    def update(self, coords: Iterable[Coord]) -> None:
        self._coords.update(coords)

    def __or__(self, other: "NeighborSet") -> "NeighborSet":
        return NeighborSet(self._coords | other._coords)

    def __iter__(self):
        return iter(self._coords)

    def to_json(self) -> str:
        """Retorna string JSON serializável."""
        return json.dumps(sorted(list(self._coords)))

    def __repr__(self):
        return f"NeighborSet({len(self._coords)} items)"

    def __bool__(self):
        return bool(self._coords)

class VitalSigns:
    """Encapsula sinais vitais e permite serialização JSON legível."""

    def __init__(self, values=None):
        # aceita lista, string JSON ou None
        if isinstance(values, str):
            try:
                self._values = json.loads(values)
            except Exception:
                self._values = []
        elif isinstance(values, (list, tuple)):
            self._values = [float(v) for v in values]
        else:
            self._values = []

    def add(self, v):
        self._values.append(float(v))

    def extend(self, vs):
        self._values.extend(float(x) for x in vs)

    def to_json(self):
        """Retorna string JSON."""
        return json.dumps(self._values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        return self._values[idx]

    def __repr__(self):
        return f"VitalSigns({self._values})"

    def as_list(self):
        return list(self._values)



# ================================================================
# Célula do mapa
# ================================================================
@dataclass
class CellInfo:
    # Estado básico
    status: Status = "unknown"              # "unknown" | "clear" | "wall" | "out_of_bounds"
    floor_factor: Optional[float] = None    # dificuldade do piso (se medido)
    visited: bool = False
    last_seen_step: int = -1
    neighbors_clear: NeighborSet = field(default_factory=NeighborSet)

    # Vítima
    victim_present: bool = False
    victim_id: Optional[int] = None
    vitals_read: bool = False
    read_step: Optional[int] = None
    vitals_raw: Optional[VitalSigns] = field(default_factory=VitalSigns)       # agora vai para o CSV

    # Rastreabilidade
    discovered_by: Optional[str] = None

    # Planejamento / custo
    g_cost: Optional[float] = None
    parent: Optional[Coord] = None


# ================================================================
# Cabeçalho fixo — AGORA inclui vitals_raw
# ================================================================
CSV_HEADER = [
    "agent", "x", "y", "status", "visited", "last_seen_step",
    "floor_factor", "victim_present", "victim_id", "vitals_read", "read_step",
    "vitals_raw", "g_cost", "parent_x", "parent_y", "neighbors_clear"
]

# ================================================================
# Funções utilitárias
# ================================================================
def _row_from_cell(agent: str, xy: Coord, cell: CellInfo) -> list:
    x, y = xy
    px, py = (cell.parent if cell.parent is not None else (None, None))
    return [
        agent,
        x, y,
        cell.status,
        int(cell.visited),
        (None if cell.last_seen_step is None else int(cell.last_seen_step)),
        (None if cell.floor_factor is None else float(cell.floor_factor)),
        int(cell.victim_present),
        (None if cell.victim_id is None else int(cell.victim_id)),
        int(cell.vitals_read),
        (None if cell.read_step is None else int(cell.read_step)),
        cell.vitals_raw.to_json() if cell.vitals_raw else "[]",
        (None if cell.g_cost is None else float(cell.g_cost)),
        px, py,
        cell.neighbors_clear.to_json() if cell.neighbors_clear else "[]",
    ]


def iter_csv_rows(agent: str, grid: MapGrid) -> Iterable[list]:
    """Gera as linhas do CSV (com todos os campos padronizados)."""
    for xy, cell in grid.items():
        if cell.status != "unknown":
            yield _row_from_cell(agent, xy, cell)


def write_map_csv(filepath: str, agent_name: str, grid: MapGrid) -> None:
    """Escreve o CSV completo (agora incluindo vitals_raw e vizinhos)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for row in iter_csv_rows(agent_name, grid):
            w.writerow(row)
    print(f"[SAVE] {agent_name}: mapa salvo com {sum(1 for _ in iter_csv_rows(agent_name, grid))} células em {filepath}")


# ================================================================
# Funções de registro usadas pelo agente
# ================================================================
def record_cell(
    agent: str,
    grid: MapGrid,
    xy: Coord,
    status: str,
    step: int,
    floor_factor: Optional[float] = None,
    g_cost: Optional[float] = None,
    parent: Optional[Coord] = None,
) -> None:
    """Atualiza a célula com status e metadados."""
    cell = grid.get(xy, CellInfo())
    cell.status = status
    cell.visited = True
    cell.last_seen_step = step
    cell.discovered_by = cell.discovered_by or agent
    if floor_factor is not None:
        cell.floor_factor = float(floor_factor)
    if g_cost is not None:
        cell.g_cost = float(g_cost)
    if parent is not None:
        cell.parent = parent
    grid[xy] = cell


def record_neighbors(
    agent: str,
    grid: MapGrid,
    xy: Coord,
    neighbors: Dict[Coord, str],
    step: int,
) -> None:
    """Marca vizinhos e atualiza lista de clear."""
    for n_xy, n_status in neighbors.items():
        cell = grid.get(n_xy, CellInfo())
        cell.status = n_status
        cell.discovered_by = cell.discovered_by or agent
        cell.last_seen_step = step
        grid[n_xy] = cell

    current = grid.get(xy, CellInfo())
    clear_coords = [pos for pos, st in neighbors.items() if st == "clear"]
    current.neighbors_clear.update(clear_coords)
    grid[xy] = current


def record_victim(
    agent: str,
    grid: MapGrid,
    xy: Coord,
    victim_id: int,
    vitals_read: bool,
    step: int,
    vitals_raw: Optional["VitalSigns"] = None,  # ✅ agora usa a classe correta
) -> None:
    """
    Registra a presença de uma vítima em (x, y) e, se aplicável,
    armazena os sinais vitais coletados.
    """
    cell = grid.get(xy, CellInfo())
    cell.victim_present = True
    cell.victim_id = victim_id
    cell.discovered_by = cell.discovered_by or agent

    if vitals_read:
        cell.vitals_read = True
        cell.read_step = step

        if vitals_raw is not None:
            cell.vitals_raw = vitals_raw
        else:
            # garante compatibilidade retroativa
            cell.vitals_raw = None

    grid[xy] = cell


def read_map_csv(filepath: str) -> MapGrid:
    """
    Lê um arquivo CSV de mapa salvo e reconstrói o dicionário MapGrid,
    com cada célula sendo uma instância completa de CellInfo.

    Retorna:
        MapGrid: dict[(x, y)] = CellInfo
    """
    import csv
    import json

    grid: MapGrid = {}

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                x, y = int(row["x"]), int(row["y"])
            except (KeyError, ValueError):
                continue  # ignora linha inválida

            cell = CellInfo()

            # Preenchimento básico
            cell.status = row.get("status", "unknown")
            cell.visited = bool(int(row.get("visited", 0)))
            cell.last_seen_step = (
                int(row["last_seen_step"]) if row.get("last_seen_step") not in ("", "None", None) else -1
            )

            # Campos numéricos
            ff = row.get("floor_factor")
            cell.floor_factor = float(ff) if ff not in ("", "None", None) else None

            gc = row.get("g_cost")
            cell.g_cost = float(gc) if gc not in ("", "None", None) else None

            # Parent
            try:
                px = int(row["parent_x"]) if row.get("parent_x") not in ("", "None", None) else None
                py = int(row["parent_y"]) if row.get("parent_y") not in ("", "None", None) else None
                if px is not None and py is not None:
                    cell.parent = (px, py)
            except Exception:
                pass

            # Vítima
            cell.victim_present = bool(int(row.get("victim_present", 0)))
            cell.victim_id = (
                int(row["victim_id"]) if row.get("victim_id") not in ("", "None", None) else None
            )
            cell.vitals_read = bool(int(row.get("vitals_read", 0)))
            cell.read_step = (
                int(row["read_step"]) if row.get("read_step") not in ("", "None", None) else None
            )

            # vitals_raw (classe VitalSigns)
            vraw = row.get("vitals_raw")
            if vraw and vraw not in ("None", ""):
                cell.vitals_raw = VitalSigns(vraw)

            # neighbors_clear (classe NeighborSet)
            neigh_json = row.get("neighbors_clear")
            if neigh_json and neigh_json not in ("None", ""):
                try:
                    ncoords = json.loads(neigh_json)
                    cell.neighbors_clear = NeighborSet(ncoords)
                except Exception:
                    cell.neighbors_clear = NeighborSet()

            grid[(x, y)] = cell

    print(f"[LOAD] Mapa reconstruído de {filepath} ({len(grid)} células)")
    return grid
