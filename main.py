import os
from agents.explorer_agent import ExplorerAgent
from vs.constants import VS
from vs.environment import Env
import config as cfg


def create_configs():
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


if __name__ == "__main__":
    # 🔹 Cria configs e ambiente
    create_configs()

    env = Env(vict_folder=cfg.VICT_FOLDER, env_folder=cfg.ENV_FOLDER)

    # 🔹 Cria agentes exploradores
    ex_agents = [
        ExplorerAgent(
            env, f"{cfg.AGENT_CONFIG_FOLDER}/explorer_{i + 1}.txt", seed=42 + i
        )
        for i in range(cfg.N_EXPLORER_AGENTS)
    ]

    # 🔹 Ativa agentes
    for agent in ex_agents:
        agent.set_state(VS.ACTIVE)

    # 🔹 Executa simulação
    env.run()

    # --- 🔸 Salvamento forçado dos mapas (novo) ---
    from agents.map_structures import write_map_csv

    os.makedirs("outputs", exist_ok=True)

    for agent in ex_agents:
        if hasattr(agent, "grid"):
            csv_path = os.path.join("outputs", f"map_explorer_{agent.name}.csv")
            write_map_csv(csv_path, agent.name, agent.grid)
            print(f"[FORCED SAVE] {agent.name}: mapa salvo em {csv_path} ({len(agent.grid)} células)")

    # --- 🧩 Etapa 2: Unificação automática dos mapas ---
    try:
        from analysis.merge_maps import unify_all_maps
        print("\n[UNIFY] Iniciando unificação dos mapas...\n")
        unify_all_maps("outputs")
    except Exception as e:
        print(f"[UNIFY] Erro durante unificação: {e}")

    # 🔹 Resultados originais (mantidos)
    for agent in ex_agents:
        agent.save_results(f"teste/victims_found_{agent.NAME}.txt")

    env.print_results()
    env.print_acum_results()
