"""Entry point for the disaster response multi-agent system."""
from __future__ import annotations

import argparse
from pathlib import Path

from vs.environment import Env

from .coordinator import MissionCoordinator
from .explorer import ExplorerDFS
from .ml_models import VictimModels
from .rescuer import RescuerMaster, RescuerWorker


def build_default_paths() -> dict[str, Path]:
    base_dir = Path(__file__).resolve().parents[2]
    return {
        "victims": base_dir / "datasets" / "vict" / "408v",
        "environment": base_dir / "datasets" / "env" / "94x94_408v",
        "config": Path(__file__).resolve().parent / "config" / "tlim1000",
        "output": Path(__file__).resolve().parent / "output",
    }


def parse_args() -> argparse.Namespace:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(description="Sistema multiagente para exploração e agrupamento de vítimas")
    parser.add_argument("--victims", type=Path, default=defaults["victims"], help="Pasta com o dataset de sinais vitais")
    parser.add_argument("--environment", type=Path, default=defaults["environment"], help="Pasta com os arquivos do ambiente")
    parser.add_argument("--config", type=Path, default=defaults["config"], help="Diretório com as configurações dos agentes")
    parser.add_argument("--output", type=Path, default=defaults["output"], help="Diretório de saída para os arquivos de cluster")
    parser.add_argument("--clusters", type=int, default=3, help="Número de clusters atribuídos pelo socorrista mestre")
    return parser.parse_args()


def load_config_files(config_dir: Path, subdir: str, prefix: str) -> list[Path]:
    target = config_dir / subdir
    files = sorted(target.glob(f"{prefix}_*.txt"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo de configuração encontrado em {target} com prefixo {prefix}")
    return files


def main() -> None:
    args = parse_args()
    env = Env(str(args.victims), str(args.environment))

    models = VictimModels(args.victims / "data.csv")
    model_metrics = {
        "tri_accuracy": models.metrics.tri_accuracy,
        "sobr_mae": models.metrics.sobr_mae,
    }
    coordinator = MissionCoordinator(
        total_explorers=3,
        nb_clusters=args.clusters,
        output_dir=args.output,
        model_metrics=model_metrics,
    )

    explorer_files = load_config_files(args.config, "explorers", "explorer")
    rescuer_files = load_config_files(args.config, "rescuers", "rescuer")

    if len(explorer_files) < 3 or len(rescuer_files) < 3:
        raise ValueError("É necessário fornecer pelo menos três arquivos de configuração para exploradores e socorristas.")

    master_file = rescuer_files[0]
    worker_files = rescuer_files[1:]
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    master = RescuerMaster(env, str(master_file), coordinator, output_dir=output_dir, nb_clusters=args.clusters)
    workers = [RescuerWorker(env, str(path), coordinator) for path in worker_files]

    # Explorers with distinct direction preferences to reduce overlap
    direction_preferences = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7, 0, 1],
        [6, 7, 0, 1, 2, 3, 4, 5],
    ]
    for file_path, preference in zip(explorer_files, direction_preferences):
        ExplorerDFS(env, str(file_path), coordinator, models, direction_preference=preference)

    master.workers = workers  # type: ignore[attr-defined]

    env.run()


if __name__ == "__main__":
    main()
