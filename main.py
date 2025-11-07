import os
from agents.explorer_agent import ExplorerAgent
from vs.constants import VS
from vs.environment import Env
from analysis.statistics import overlap_metric
from analysis.cluster_victims import main as run_cluster
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

    # 🔸 Remove agentes mortos antes de salvar ou analisar
    alive_agents = [a for a in ex_agents if a.get_state() != VS.DEAD]

    if not alive_agents:
        print("\n⚠️ Todos os exploradores morreram antes do fim da missão! Nenhum resultado válido.")
    else:
        print(f"\n[STATUS] {len(alive_agents)} exploradores ainda vivos no final da missão:")
        for a in alive_agents:
            state = a.get_state()
            state_name = state.name if hasattr(state, "name") else state
            print(f"  - {a.name} ({state_name})")


    # --- 🔸 Salvamento forçado dos mapas (novo) ---
    from agents.map_structures import write_map_csv

    os.makedirs("outputs", exist_ok=True)

    for agent in alive_agents:
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

     # --- 🎨 Exibir mapa unificado com triagem ---
    try:


        tri_colors = {
            "green": (0, 1, 0),
            "yellow": (1, 1, 0),
            "red": (1, 0, 0),
            "black": (0.2, 0.2, 0.2),
        }

        print("\n[VISUALIZER] Exibindo mapa unificado com cores de triagem...\n")

    except Exception as e:
        print(f"[VISUALIZER] Erro ao exibir mapa unificado colorido: {e}")

    # 🔹 Resultados originais (mantidos)
    for agent in alive_agents:
        agent.save_results(f"teste/victims_found_{agent.NAME}.txt")

    env.print_results()
    env.print_acum_results()

    try:
        print("\n=== ETAPA 3: Estatísticas e Clustering ===")
        # Estatísticas
        per_agent, ve, s = overlap_metric()
        print(f"\n[STATS] Vítimas por explorador: {per_agent}")
        print(f"[STATS] Vítimas únicas (Ve): {ve}")
        if s is not None:
            print(f"[STATS] Sobreposição: {s:.4f}")
        else:
            print("[STATS] Sobreposição: N/A")

        # Clustering
        run_cluster(k=3)

    except Exception as e:
        print("\n[ERRO] Falha ao executar estatísticas ou clustering:")
        import traceback
        traceback.print_exc()
