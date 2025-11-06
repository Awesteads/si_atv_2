import os
from typing import List

from agents.explorer_agent import ExplorerAgent
from agents.map_sync import MapSynchronizer
from agents.rescuer_agent import RescuerAgent
from vs.constants import VS
from vs.environment import Env

import config as cfg


def create_explorer_configs():
    """Create/Overwrite config files for explorer agents."""
    os.makedirs(cfg.AGENT_CONFIG_FOLDER, exist_ok=True)
    for i in range(1, cfg.N_EXPLORER_AGENTS + 1):
        config_path = os.path.join(cfg.AGENT_CONFIG_FOLDER, f"explorer_{i}.txt")
        with open(config_path, "w") as f:
            f.write(
                f"NAME EXPLORER_{i}\n"
                "COLOR (0, 255, 0)\n"
                "TRACE_COLOR (0, 100, 0)\n"
                f"TLIM {cfg.TIME_EXPLORER_LIMIT}\n"
                f"COST_LINE {cfg.COST_EXPLORER_LINE}\n"
                f"COST_DIAG {cfg.COST_EXPLORER_DIAG}\n"
                f"COST_READ {cfg.COST_EXPLORER_READ}\n"
                f"COST_FIRST_AID {cfg.COST_EXPLORER_FIRST_AID}\n"
            )


def create_rescuer_configs():
    """Create/Overwrite config files for rescuer agents."""
    os.makedirs(cfg.RESCUER_CONFIG_FOLDER, exist_ok=True)
    for i in range(1, cfg.N_RESCUER_AGENTS + 1):
        config_path = os.path.join(cfg.RESCUER_CONFIG_FOLDER, f"rescuer_{i}.txt")
        with open(config_path, "w") as f:
            f.write(
                f"NAME RESCUER_{i}\n"
                "COLOR (255, 0, 0)\n"
                "TRACE_COLOR (120, 0, 0)\n"
                f"TLIM {cfg.TIME_RESCUER_LIMIT}\n"
                f"COST_LINE {cfg.COST_RESCUER_LINE}\n"
                f"COST_DIAG {cfg.COST_RESCUER_DIAG}\n"
                f"COST_READ {cfg.COST_RESCUER_READ}\n"
                f"COST_FIRST_AID {cfg.COST_RESCUER_FIRST_AID}\n"
            )


if __name__ == "__main__":
    os.makedirs("teste", exist_ok=True)

    create_explorer_configs()

    env = Env(vict_folder=cfg.VICT_FOLDER, env_folder=cfg.ENV_FOLDER)

    ex_agents = [
        ExplorerAgent(
            env, f"{cfg.AGENT_CONFIG_FOLDER}/explorer_{i + 1}.txt", seed=42 + i
        )
        for i in range(cfg.N_EXPLORER_AGENTS)
    ]

    for agent in ex_agents:
        agent.set_state(VS.ACTIVE)
    env.run()

    surviving_maps = []
    for agent in ex_agents:
        if agent.get_state() != VS.DEAD:
            agent.save_results(f"teste/victims_found_{agent.NAME}.txt")
            surviving_maps.append(agent.get_local_map())

    synchronizer = MapSynchronizer(
        grid_size=(env.dic["GRID_WIDTH"], env.dic["GRID_HEIGHT"]),
        base=tuple(env.dic["BASE"]),
    )
    unified_map = synchronizer.merge(surviving_maps)
    synchronizer.save_to_json(unified_map, "teste/unified_map.json")
    synchronizer.save_summary(unified_map, "teste/unified_map_summary.txt")
    print(synchronizer.build_summary(unified_map))

    env.print_results()
    env.print_acum_results()

    if unified_map.victims and cfg.N_RESCUER_AGENTS > 0:
        create_rescuer_configs()

        rescue_env = Env(vict_folder=cfg.VICT_FOLDER, env_folder=cfg.ENV_FOLDER)

        rescuer_agents: List[RescuerAgent] = [
            RescuerAgent(
                rescue_env,
                os.path.join(cfg.RESCUER_CONFIG_FOLDER, f"rescuer_{i + 1}.txt"),
                unified_map,
            )
            for i in range(cfg.N_RESCUER_AGENTS)
        ]

        for agent in rescuer_agents:
            agent.set_state(VS.ACTIVE)

        rescue_env.run()

        for agent in rescuer_agents:
            agent.save_results(f"teste/rescue_report_{agent.NAME}.txt")

        rescue_env.print_results()
        rescue_env.print_acum_results()
    else:
        print("No victims available for the rescue phase or no rescuers configured.")
