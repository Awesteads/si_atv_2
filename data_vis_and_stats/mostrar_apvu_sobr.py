from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =============================
# CONFIGURAÇÕES DE PASTA
# =============================


BASE = Path("../datasets/vict/100v")
INPUT = BASE / "data.csv"
OUTDIR = BASE / "relatorios"
FIGDIR = OUTDIR / "figs"
ORDER = ["A", "V", "P", "U"]

AVPU_MAP = {0: "A", 1: "V", 2: "P", 3: "U"}

# =============================
# FUNÇÕES AUXILIARES
# =============================
def correlation_ratio(categories, values):
    categories = np.asarray(categories)
    values = np.asarray(values, dtype=float)
    mask = ~pd.isna(categories) & ~pd.isna(values)
    categories, values = categories[mask], values[mask]
    if len(values) == 0:
        return np.nan
    levels = [values[categories == k] for k in np.unique(categories)]
    mean_total = values.mean()
    ss_between = sum(len(x) * (x.mean() - mean_total) ** 2 for x in levels if len(x) > 0)
    ss_total = ((values - mean_total) ** 2).sum()
    return np.sqrt(ss_between / ss_total) if ss_total > 0 else np.nan

def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    m = x.mean()
    sd = x.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else np.nan
    tcrit = stats.t.ppf(1 - 0.025, df=max(n - 1, 1)) if n > 1 else np.nan
    half = tcrit * se if n > 1 else np.nan
    lo = m - half if n > 1 else np.nan
    hi = m + half if n > 1 else np.nan
    return m, sd, lo, hi

# =============================
# EXECUÇÃO PRINCIPAL
# =============================
def main():
    if not INPUT.exists():
        raise SystemExit(f"Arquivo não encontrado: {INPUT}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT, encoding="utf-8")
    if "avpu" not in df.columns or "sobr" not in df.columns:
        raise SystemExit("O dataset precisa conter as colunas 'avpu' e 'sobr'.")

    # Converte codificação inteira para rótulos
    avpu = df["avpu"].map(AVPU_MAP)
    sobr = pd.to_numeric(df["sobr"], errors="coerce")
    df2 = pd.DataFrame({"avpu": avpu, "sobr": sobr})
    df2 = df2.dropna(subset=["sobr"])

    total_rows = len(df2)
    unexpected = sorted(set(df2["avpu"].dropna()) - set(ORDER))
    na_count = int(df["avpu"].isna().sum())

    df_ok = df2[df2["avpu"].isin(ORDER)].copy()
    df_ok["avpu"] = pd.Categorical(df_ok["avpu"], categories=ORDER, ordered=True)

    stats_rows = []
    for lvl in ORDER:
        vals = df_ok.loc[df_ok["avpu"] == lvl, "sobr"].values
        n = len(vals)
        m, sd, lo, hi = mean_ci95(vals)
        stats_rows.append({"avpu": lvl, "N": n, "mean": m, "std": sd, "ci95_lo": lo, "ci95_hi": hi})
    stats_df = pd.DataFrame(stats_rows)

    present_levels = [df_ok.loc[df_ok["avpu"] == lvl, "sobr"].values for lvl in ORDER if (df_ok["avpu"] == lvl).any()]
    eta = correlation_ratio(df_ok["avpu"], df_ok["sobr"])
    if all(len(v) >= 2 for v in present_levels) and len(present_levels) >= 2:
        f_stat, p_val = stats.f_oneway(*present_levels)
    else:
        f_stat, p_val = np.nan, np.nan

    # -----------------------------
    # FIGURA 1: Boxplot + dispersão
    # -----------------------------
    plt.figure(figsize=(8, 5))
    positions = np.arange(1, len(ORDER) + 1)
    data_for_plot = [df_ok.loc[df_ok["avpu"] == lvl, "sobr"].values for lvl in ORDER]
    bp = plt.boxplot(data_for_plot, positions=positions, widths=0.6, patch_artist=True)
    for i, vals in enumerate(data_for_plot, start=1):
        if len(vals) == 0:
            continue
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.15
        plt.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.5, s=18)
    plt.xticks(positions, ORDER)
    plt.ylabel("sobr")
    plt.title(f"AVPU × SOBR  |  eta={eta:.3f}  p(ANOVA)={p_val:.3g}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig1_path = FIGDIR / "avpu_sobr_box_scatter.png"
    plt.savefig(fig1_path, dpi=150)
    plt.close()

    # -----------------------------
    # FIGURA 2: Médias com IC95%
    # -----------------------------
    plt.figure(figsize=(8, 4.5))
    means = stats_df["mean"].values
    errs_lo = means - stats_df["ci95_lo"].values
    errs_hi = stats_df["ci95_hi"].values - means
    errs_lo = np.nan_to_num(errs_lo, nan=0.0)
    errs_hi = np.nan_to_num(errs_hi, nan=0.0)
    yerr = np.vstack([errs_lo, errs_hi])
    plt.errorbar(ORDER, means, yerr=yerr, fmt="o-", capsize=4)
    plt.ylabel("sobr (média ± IC95%)")
    plt.title("Médias de SOBR por AVPU")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2_path = FIGDIR / "avpu_sobr_means_ci.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close()

    # -----------------------------
    # MARKDOWN
    # -----------------------------
    lines = []
    lines.append("# AVPU × SOBR\n")
    lines.append(f"- **Arquivo**: `{INPUT.name}`")
    lines.append(f"- **Linhas válidas p/ análise**: {len(df_ok)}\n")
    lines.append(f"- **Eta (razão de correlação)**: {eta:.4f}\n")
    lines.append(f"  Eta mede quanto da variabilidade total de uma variável numérica pode ser explicada por uma variável categórica.\n")
    lines.append(f"  Valores próximos de 0 indicam associação fraca e quanto mais próximos de 1, mais forte.")
    lines.append(f"- **ANOVA**: F={f_stat:.4f}, p={p_val:.4g}\n")
    lines.append(f"  Mede se as médias da variável numérica são significativamente diferentes para os valores da variável categórica.\n")
    lines.append(f"  p < 0,05 → rejeita-se H₀ → existe diferença estatisticamente significativa entre as médias.\n")
    lines.append(f"  p ≥ 0,05 → não há evidência suficiente para rejeitar H₀ → médias podem ser consideradas estatisticamente iguais.")
    if na_count > 0:
        lines.append(f"- **Ausentes em AVPU**: {na_count}")
    if unexpected:
        lines.append(f"- **Valores inesperados em AVPU**: {', '.join(map(str, unexpected))}")
    lines.append("\n## Estatísticas por categoria")
    tbl = stats_df.copy()
    for col in ["mean", "std", "ci95_lo", "ci95_hi"]:
        tbl[col] = tbl[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    lines.append(tbl.to_markdown(index=False))
    lines.append("\n## Gráficos")
    lines.append(f"![Boxplot e dispersão](figs/{fig1_path.name})")
    lines.append(f"![Médias com IC95%](figs/{fig2_path.name})\n")

    (OUTDIR / "summary_avpu_sobr.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"OK: relatório salvo em {OUTDIR}")

if __name__ == "__main__":
    main()
