
"""
=====================================================================
Correlação (Pearson) com comutação automática para LOWESS
=====================================================================

O script:
1) Lê o dataset de sinais vitais (vars. BASE e INPUT)
2) Para cada par (fc, fr, pas, spo2, gcs) × (sobr):
   - Calcula r e p (Pearson)
   - Se p >= 0.05 ou |r| < 0.8 → plota DISPERSÃO com LOWESS
   - Caso contrário → regressão linear
3) Salva PNGs individuais e um PNG com todos os gráficos.

Dependências: pandas, numpy, matplotlib, seaborn
LOWESS: requer statsmodels (opcional; se ausente, cai em fallback sem linha)
=====================================================================
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import os

# ------------------------
# Configurações de Paths
# ------------------------
BASE = Path("../datasetS/vict/100v")
INPUT = BASE / "data.csv"
OUTDIR = BASE / "relatorios"
FIGDIR    = OUTDIR / "figs"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

# Pares de variáveis
VAR_PAIRS = [
    ("fc", "sobr"),
    ("fr", "sobr"),
    ("pas", "sobr"),
    ("spo2", "sobr"),
    ("gcs", "sobr"),
]

ALPHA = 0.05
R_THRESHOLD = 0.8

# ------------------------
# Funções auxiliares
# ------------------------
def fmt_p(p):
    if np.isnan(p):
        return "na"
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"

def safe_pearson(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan, n
    r, p = pearsonr(x, y)
    return r, p, n

def should_use_lowess(r, p, alpha=ALPHA, r_thresh=R_THRESHOLD):
    if np.isnan(r) or np.isnan(p):
        return True
    return (p >= alpha) or (abs(r) < r_thresh)

def annotate(ax, r, p, n, mode):
    ax.text(
        0.03, 0.97, f"r = {r:.3f}\np = {fmt_p(p)}\nN = {n}\nmode = {mode}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.75)
    )

def regplot_auto(ax, data, x, y, use_lowess):
    scatter_kws = {"alpha": 0.6}
    line_kws = {"color": "red"}
    if use_lowess:
        try:
            sns.regplot(x=x, y=y, data=data, ax=ax, lowess=True,
                        scatter_kws=scatter_kws, line_kws=line_kws)
            return "LOWESS"
        except Exception:
            sns.scatterplot(x=x, y=y, data=data, ax=ax, alpha=0.6)
            return "LOWESS (indisp.)"
    else:
        sns.regplot(x=x, y=y, data=data, ax=ax, lowess=False,
                    scatter_kws=scatter_kws, line_kws=line_kws)
        return "Linear"

# ------------------------
# Execução Principal
# ------------------------
def main():
    if not INPUT.exists():
        raise SystemExit(f"Arquivo não encontrado: {INPUT}")

    df = pd.read_csv(INPUT, encoding="utf-8")
    md_file = OUTDIR / "summary_correlations.md"

    # Início do relatório
    lines = []
    lines.append("# Relatório de Correlações (Pearson) com LOWESS\n")
    lines.append(
        "Este relatório apresenta a análise de correlação entre variáveis numéricas usando o **coeficiente de Pearson**.\n"
        "- O **coeficiente de Pearson (r)** mede a relação linear entre duas variáveis.\n"
        "- O **LOWESS com comutação automática** desenha uma curva suave para mostrar tendências não-lineares quando a correlação é fraca ou p-valor alto.\n"
        "- Se a correlação é forte e significativa, usa-se regressão linear para a linha de tendência.\n"
    )
    lines.append("\n---\n")

    for x, y in VAR_PAIRS:
        if x not in df.columns or y not in df.columns:
            print(f"Colunas {x} ou {y} não encontradas.")
            continue

        data = df[[x, y]].dropna()
        r, p, n = safe_pearson(data[x], data[y])
        use_low = should_use_lowess(r, p)
        mode = "LOWESS" if use_low else "Linear"

        # Gráfico individual
        fig_ind, ax_ind = plt.subplots(figsize=(6, 4))
        mode_ind = regplot_auto(ax_ind, data, x, y, use_low)
        annotate(ax_ind, r, p, n, mode_ind)
        ax_ind.set_title(f"{x} × {y}")
        ax_ind.grid(True, linestyle="--", alpha=0.6)
        fig_ind.tight_layout()
        ind_png = FIGDIR / f"correlacao_{x}_{y}.png"
        fig_ind.savefig(ind_png, dpi=150)
        plt.close(fig_ind)

        # Adiciona ao Markdown
        lines.append(f"## {x} × {y}\n")
        lines.append(f"- r = {r:.3f}, p = {fmt_p(p)}, N = {n}, modo = {mode_ind}\n")
        lines.append(f"![{x} × {y}](figs/{ind_png.name})\n")
        lines.append("\n---\n")

    # Salvar Markdown
    md_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Relatório salvo em: {md_file}")

if __name__ == "__main__":
    main()
