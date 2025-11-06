# Configuração do Agente

Cada agente deve ter um arquivo de configuração.

---

## Parâmetros

| Propriedade        | Valor             | Descrição                                                |
|------------------|-------------------|----------------------------------------------------------|
| NAME             | EXPL_1            | Nome do agente                                           |
| COLOR            | (103, 103, 255)   | Cor principal do agente (RGB)                            |
| TRACE_COLOR      | (103, 103, 255)   | Cor da trilha deixada pelo agente (RGB)                  |
| TLIM             | 5000              | Limite de tempo para exploração ou socorro                |
| COST_LINE        | 1.0               | Custo de movimento em linha reta                         |
| COST_DIAG        | 1.5               | Custo de movimento em diagonal                           |
| COST_READ        | 2.0               | Custo para ação de leitura dos sinais vitais      |
| COST_FIRST_AID   | 1.0               | Custo para prestar primeiros socorros a uma vítima       |

---
