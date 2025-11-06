from pathlib import Path
import pandas as pd
import numpy as np

# === Configurações fixas ===
BASE_FOLDER = Path("../datasets/vict/100v")
INPUT_CSV   = BASE_FOLDER / "data.csv"
OUTPUT_MD   = BASE_FOLDER / "summary.md"

# Esquema conhecido (ordem preservada nos conjuntos)
CAT_DOMAINS = {
    "pr":    [0, 1],
    "queim": [0, 1, 2, 3],
    "fx":    [0, 1],
    "sg":    [0, 1, 2, 3],
    "avpu":  [0, 1, 2, 3],
    "tri":   [0, 1, 2, 3],  
}

MAP_SN     = ['N', 'S']
MAP_LEVELS = ['NAO', 'LEVE', 'MODERADO', 'GRAVE']
MAP_AVPU   = ['A', 'V', 'P', 'U']
MAP_TRI    = ['VERDE', 'AMARELO', 'VERMELHO', 'PRETO']

NUM_INT_COLS   = ["idade", "fc", "fr", "pas", "spo2", "gcs"]
NUM_FLOAT_COLS = ["sobr"]  # [0,1]

# ------------------------
# Funções auxiliares
# ------------------------
def read_and_cast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")

    # Converte numéricos conhecidos
    for c in NUM_INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in NUM_FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Converte categóricos para int
    for c in CAT_DOMAINS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def numeric_section(df: pd.DataFrame) -> str:
    lines = []
    cols = [c for c in NUM_INT_COLS + NUM_FLOAT_COLS if c in df.columns]
    if not cols:
        return ""

    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            rows.append((c, np.nan, np.nan, np.nan))
        else:
            rows.append((c, s.min(), s.max(), s.mean()))
    out = pd.DataFrame(rows, columns=["coluna", "mín", "máx", "média"])

    # formata
    fmtd = out.copy()
    for col in ["mín", "máx", "média"]:
        fmtd[col] = fmtd[col].map(
            lambda x: "" if pd.isna(x) else (
                f"{x:.3f}" if isinstance(x, float) and not x.is_integer() else f"{int(x)}"
            )
        )

    lines.append("## Estatísticas (numéricas)")
    lines.append(fmtd.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)

def _fmt_pct(x: int, total: int) -> str:
    return f"{(100.0 * x / total):.2f}%" if total > 0 else "0.00%"

def map_label(cat_name: str, code: int) -> str:
    """Retorna o rótulo string para o código inteiro."""
    if cat_name in ["pr", "fx"]:
        return MAP_SN[code] if 0 <= code < len(MAP_SN) else f"<invalido:{code}>"
    elif cat_name in ["queim", "sg"]:
        return MAP_LEVELS[code] if 0 <= code < len(MAP_LEVELS) else f"<invalido:{code}>"
    elif cat_name == "avpu":
        return MAP_AVPU[code] if 0 <= code < len(MAP_AVPU) else f"<invalido:{code}>"
    elif cat_name == "tri":
        return MAP_TRI[code] if 0 <= code < len(MAP_TRI) else f"<invalido:{code}>"
    else:
        return str(code)

def categorical_section(df: pd.DataFrame) -> str:
    lines = []
    present_cats = [c for c in CAT_DOMAINS if c in df.columns]
    if not present_cats:
        return ""

    total_rows = len(df)
    lines.append("## Contagem por valor (categóricas)")

    for c in present_cats:
        domain = CAT_DOMAINS[c]  # ordem preservada
        s = df[c]
        vc = s.value_counts(dropna=False)  # inclui NaN

        # Tabela principal: todos valores do domínio (mesmo que zero)
        expected_rows = []
        for val in domain:
            cnt = int(vc.get(val, 0))
            label = map_label(c, val)
            expected_rows.append((val, label, cnt, _fmt_pct(cnt, total_rows)))
        expected_df = pd.DataFrame(expected_rows, columns=["código", "valor", "contagem", "%"])
        lines.append(f"### `{c}`")
        lines.append(expected_df.to_markdown(index=False))

        # Ausentes
        na_cnt = int(s.isna().sum())
        if na_cnt > 0:
            lines.append("")
            lines.append("> **Ausentes**")
            na_df = pd.DataFrame(
                [("<NA>", na_cnt, _fmt_pct(na_cnt, total_rows))],
                columns=["valor", "contagem", "%"]
            )
            lines.append(na_df.to_markdown(index=False))

        # Inesperados
        unexpected_items = [(k, int(v)) for k, v in vc.items() if pd.notna(k) and k not in domain]
        if unexpected_items:
            rows_u = [(k, map_label(c, k), v, _fmt_pct(v, total_rows)) for k, v in unexpected_items]
            u_df = pd.DataFrame(rows_u, columns=["código", "valor", "contagem", "%"])
            lines.append("")
            lines.append("> **Valores inesperados** (fora do domínio esperado)")
            lines.append(u_df.to_markdown(index=False))

        lines.append("")
    return "\n".join(lines)

def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Arquivo não encontrado: {INPUT_CSV}")

    df = read_and_cast(INPUT_CSV)

    lines = [
        "# Resumo do CSV\n",
        f"- **Arquivo**: `{INPUT_CSV.name}`",
        f"- **Número de linhas**: {len(df)}\n",
        numeric_section(df),
        categorical_section(df),
    ]

    md = "\n".join([blk for blk in lines if blk])
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"OK: relatório gerado em {OUTPUT_MD}")

if __name__ == "__main__":
    main()
