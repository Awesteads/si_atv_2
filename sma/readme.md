# Configuração do Agente

Cada agente deve ter um arquivo de configuração.

---

## Tarefa 2 – Exploração e Agrupamento

Implementação completa do cenário solicitado na tarefa 2 está disponível em `sma/tarefa2`.

### Como executar

```bash
cd sma/tarefa2
python -m tarefa2.main --config config/tlim1000
# ou para TLIM=8000
python -m tarefa2.main --config config/tlim8000
```

Parâmetros adicionais permitem alterar a pasta de sinais vitais (`--victims`),
os arquivos do ambiente (`--environment`), o diretório de saída dos arquivos de
cluster (`--output`) e o número de agrupamentos (`--clusters`).

Os arquivos `cluster*.txt` são escritos no diretório informado em `--output`.

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
