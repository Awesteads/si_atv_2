# agents/map_structures.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Set, Iterable
import csv

# Valores válidos para status: "unknown" | "clear" | "wall" | "out_of_bounds"
Status = str
Coord = Tuple[int, int]

@dataclass
class CellInfo:
    status: Status = "unknown"                  # unknown|clear|wall|out_of_bounds
    floor_factor: Optional[float] = None        # Dificuldade do piso
    visited: bool = False
    last_seen_step: int = -1
    neighbors_clear: Set[Coord] = field(default_factory=set)

    # Vítima
    victim_present: bool = False
    victim_id: Optional[int] = None
    vitals_read: bool = False
    read_step: Optional[int] = None

    # Rastreabilidade
    discovered_by: Optional[str] = None

    # Planejamento / custo
    g_cost: Optional[float] = None
    parent: Optional[Coord] = None

MapGrid = Dict[Coord, CellInfo]

# Cabeçalho padronizado para os CSVs de cada agente
CSV_HEADER = [
    "agent", "x", "y", "status", "visited", "last_seen_step",
    "floor_factor", "victim_present", "victim_id", "vitals_read", "read_step",
    "g_cost", "parent_x", "parent_y"
]

def _row_from_cell(agent: str, xy: Coord, cell: CellInfo) -> list:
    x, y = xy
    px, py = (cell.parent if cell.parent is not None else (None, None))
    return [
        agent,
        x, y,
        cell.status,
        int(cell.visited),
        cell.last_seen_step,
        (None if cell.floor_factor is None else float(cell.floor_factor)),
        int(cell.victim_present),
        (None if cell.victim_id is None else int(cell.victim_id)),
        int(cell.vitals_read),
        (None if cell.read_step is None else int(cell.read_step)),
        (None if cell.g_cost is None else float(cell.g_cost)),
        px, py,
    ]

def iter_csv_rows(agent: str, grid: MapGrid) -> Iterable[list]:
    """Gera as linhas (inclui apenas células conhecidas, ou seja, status != 'unknown')."""
    for xy, cell in grid.items():
        if cell.status != "unknown":
            yield _row_from_cell(agent, xy, cell)

def write_map_csv(filepath: str, agent: str, grid: MapGrid) -> None:
    """Salva o mapa local do agente em CSV no formato padronizado."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for row in iter_csv_rows(agent, grid):
            w.writerow(row)

# ---------------------------------------------------------------------------
# Funções auxiliares de registro para uso pelos agentes
# ---------------------------------------------------------------------------

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
    """
    Atualiza ou cria a entrada de uma célula no mapa local do agente.
    """
    cell = grid.get(xy, CellInfo())
    cell.status = status
    cell.visited = True
    cell.last_seen_step = step
    cell.discovered_by = cell.discovered_by or agent

    if floor_factor is not None:
        cell.floor_factor = floor_factor
    if g_cost is not None:
        cell.g_cost = g_cost
    if parent is not None:
        cell.parent = parent

    grid[xy] = cell


def record_neighbors(agent: str, grid: MapGrid, xy: Coord, neighbors: Dict[Coord, str], step: int) -> None:
    """
    Marca os vizinhos observáveis como clear/wall/out_of_bounds conforme leitura.
    Exemplo: neighbors = {(x+1, y): 'clear', (x, y-1): 'wall'}
    """
    for n_xy, n_status in neighbors.items():
        cell = grid.get(n_xy, CellInfo())
        cell.status = n_status
        cell.discovered_by = cell.discovered_by or agent
        cell.last_seen_step = step
        grid[n_xy] = cell

    # também atualiza a célula atual com as coordenadas dos vizinhos clear
    current = grid.get(xy, CellInfo())
    clear_coords = {pos for pos, st in neighbors.items() if st == "clear"}
    current.neighbors_clear |= clear_coords
    grid[xy] = current


def record_victim(
    agent: str,
    grid: MapGrid,
    xy: Coord,
    victim_id: int,
    vitals_read: bool,
    step: int,
) -> None:
    """
    Registra a presença de uma vítima em (x,y) e, se aplicável, a leitura dos sinais vitais.
    """
    cell = grid.get(xy, CellInfo())
    cell.victim_present = True
    cell.victim_id = victim_id
    cell.discovered_by = cell.discovered_by or agent
    if vitals_read:
        cell.vitals_read = True
        cell.read_step = step
    grid[xy] = cell
