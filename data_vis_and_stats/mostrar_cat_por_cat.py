
"""
Análise genérica de associação entre duas variáveis categóricas

- Lê sinais vitais das vítimas (data.csv)
- Para cada par (x, y) em PAIRS:
  * Tabela de contingência (absoluta)
  * Tabela de proporções por linha (%)
  * Heatmap (Y invertido se y for 'avpu': A, V, P, U de baixo para cima)
  * Barras empilhadas (cores de TRI fixas: verde=green, amarelo=yellow, vermelho=red, preto=black)
  * Teste Qui-quadrado e Cramér’s V
  * Relatório Markdown com tabelas e métricas

Edite PAIRS abaixo para os pares desejados.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ------------------------
# CONFIGURAÇÕES
# ------------------------
BASE = Path("../datasets/vict/100v")  # Ajuste o caminho base
INPUT = os.path.join(BASE, "data.csv")
OUTDIR = os.path.join(BASE, "relatorios")
FIGDIR = os.path.join(OUTDIR, "figs")
os.makedirs(FIGDIR, exist_ok=True)

# Lista de pares de variáveis para análise
PAIRS = [
    ("avpu", "tri"),
    ("queim", "tri"),
    ("sg", "tri"),
    ("fx", "tri"),
    ("gcs", "tri"),
]

# Mapas para rótulos de variáveis categóricas
AVPU_MAP = {0: "A", 1: "V", 2: "P", 3: "U"}
TRI_MAP = {0: "Verde", 1: "Amarelo", 2: "Vermelho", 3: "Preto"}
TRI_COLORS = {"Verde": "green", "Amarelo": "yellow", "Vermelho": "red", "Preto": "black"}

# ------------------------
# LEITURA DO DATASET
# ------------------------
df = pd.read_csv(INPUT, encoding="utf-8")

# Converte colunas específicas
if "avpu" in df.columns:
    df["avpu"] = df["avpu"].map(AVPU_MAP)
if "tri" in df.columns:
    df["tri"] = df["tri"].map(TRI_MAP)

# ------------------------
# RELATÓRIO MARKDOWN
# ------------------------
relatorio_path = os.path.join(OUTDIR, "summary_cat_x_cat.md")
with open(relatorio_path, "w", encoding="utf-8") as f:

    # Introdução geral sobre o teste qui-quadrado
    f.write("# Relatório Completo de Análises\n\n")
    f.write(
        "Este relatório apresenta a análise de associação entre variáveis categóricas "
        "utilizando o **teste qui-quadrado**.\n\n"
        "- **p-valor**: probabilidade de observar os dados ou algo mais extremo se não houver associação real. "
        "Valores pequenos (ex.: < 0,05) sugerem relação significativa.\n"
        "- **Graus de liberdade (dof)**: representam quantas combinações independentes podem variar "
        "na tabela de contingência.\n\n"
    )

    # Análises para cada par
    for x, y in PAIRS:
        if x not in df.columns or y not in df.columns:
            continue

        # Tabelas de contingência
        tabela = pd.crosstab(df[x], df[y])
        tabela_prop = tabela.div(tabela.sum(axis=1), axis=0) * 100

        # Teste Qui-Quadrado
        chi2, p, dof, _ = chi2_contingency(tabela)

        # Nome base para arquivos de figuras
        base_name = f"{x}_vs_{y}"
        fig_path_heat = os.path.join(FIGDIR, f"{base_name}_heatmap.png")
        fig_path_bar = os.path.join(FIGDIR, f"{base_name}_barras.png")

        # Heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(tabela, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Heatmap de {x} x {y}")
        plt.ylabel(x)
        plt.xlabel(y)
        if x == "avpu":
            plt.gca().invert_yaxis()  # Inverte ordem para AVPU
        plt.tight_layout()
        plt.savefig(fig_path_heat, dpi=150)
        plt.close()

        # Barras empilhadas
        tabela_perc = tabela.div(tabela.sum(axis=1), axis=0)
        tabela_perc.plot(
            kind="bar",
            stacked=True,
            color=[TRI_COLORS.get(c, "gray") for c in tabela.columns],
            figsize=(6, 4)
        )
        plt.title(f"Barras Empilhadas - {x} x {y}")
        plt.ylabel("Proporção")
        plt.xlabel(x)
        plt.legend(title=y)
        plt.tight_layout()
        plt.savefig(fig_path_bar, dpi=150)
        plt.close()

        # Escreve no relatório
        f.write(f"## Análise: {x} x {y}\n\n")
        f.write("### Tabela de Contingência (Absoluta)\n\n")
        f.write(tabela.to_markdown() + "\n\n")

        f.write("### Tabela de Proporções por Linha (%)\n\n")
        f.write(tabela_prop.round(2).to_markdown() + "\n\n")

        f.write(f"### Heatmap\n\n![](figs/{base_name}_heatmap.png)\n\n")
        f.write(f"### Barras Empilhadas\n\n![](figs/{base_name}_barras.png)\n\n")

        f.write("### Teste Qui-Quadrado\n\n")
        f.write(f"- Qui2 = {chi2:.4f}\n")
        f.write(f"- p-valor = {p:.4f}\n")
        f.write(f"- Graus de liberdade = {dof}\n\n")
        f.write("---\n\n")

print(f"Relatório salvo em: {relatorio_path}")
