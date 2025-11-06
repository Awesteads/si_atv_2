# AVPU × SOBR

- **Arquivo**: `data.csv`
- **Linhas válidas p/ análise**: 100

- **Eta (razão de correlação)**: 0.9487

  Eta mede quanto da variabilidade total de uma variável numérica pode ser explicada por uma variável categórica.

  Valores próximos de 0 indicam associação fraca e quanto mais próximos de 1, mais forte.
- **ANOVA**: F=287.9134, p=7.58e-48

  Mede se as médias da variável numérica são significativamente diferentes para os valores da variável categórica.

  p < 0,05 → rejeita-se H₀ → existe diferença estatisticamente significativa entre as médias.

  p ≥ 0,05 → não há evidência suficiente para rejeitar H₀ → médias podem ser consideradas estatisticamente iguais.

## Estatísticas por categoria
| avpu   |   N |   mean |   std |   ci95_lo |   ci95_hi |
|:-------|----:|-------:|------:|----------:|----------:|
| A      |  38 |  0.91  | 0.064 |     0.889 |     0.931 |
| V      |  14 |  0.843 | 0.048 |     0.815 |     0.871 |
| P      |  30 |  0.499 | 0.147 |     0.444 |     0.554 |
| U      |  18 |  0.137 | 0.088 |     0.093 |     0.181 |

## Gráficos
![Boxplot e dispersão](figs/avpu_sobr_box_scatter.png)
![Médias com IC95%](figs/avpu_sobr_means_ci.png)
