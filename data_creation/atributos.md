# DATASET DE VÍTIMAS

---

## Variáveis geradas

| Seq|Variável  | Descrição                                             | Tipo   |
|----|----------|-------------------------------------------------------|--------|
|  1 | idade    | Idade da vítima (entre 1 e 90 anos)                   | INT    |
|  2 | fc       | Frequência cardíaca (bpm) de 0 a 160                  | INT    |
|  3 | fr       | Frequência respiratória (rpm) de 0 a 50               | INT    |
|  4 | pas      | Pressão arterial sistólica (mmHg) de 0 a 200          | INT    |                  
|  5 | spo2     | Saturação de oxigênio (%)   de 0 a 100                | INT    |     
|  6 | temp     | Temperatura corporal (°C)  de 0 a 38.5                | FLOAT  |
|  7 | pr       | pulso radial: 0 (NÃO), 1 (SIM)                        | INT    |
|  8 | sg       | sangramento: 0 (NÃO), 1 (LEVE), 2 (MODERADO), 3 (GRAVE) | INT    |
|  9 | fx       | fratura exposta: 0 (NÃO), 1 (SIM)                           | INT |
| 10 | queim    | queimadura: 0 (NÃO), 1 (LEVE), 2 (MODERADO), 3 (GRAVE) | INT |
| 11 | gcs      | [Escala de Coma de Glasgow (3–15)](https://pt.wikipedia.org/wiki/Escala_de_coma_de_Glasgow) |   INT |
| 12 | avpu     | Estado de consciência: 0 (ALERTA), 1 (VOZ), 2 (PAIN), 3 (INCONSCIENTE) | INT |
| 13 | tri      | Triagem classif. START: 0 (verde), 1 (amarelo), 2 (vermelho), 3 (preto) | INT |
| 14 | sobr     | Prob. de sobrevivência (0 a 1)  REAL                    | FLOAT |

<small>
APVU: se a vítima está alerta (A), responde a estímulos de voz (V), de dor (P) ou se está inconsciente (U).

GCS: de 3 a 15, quanto menor o valor, menos grave a situação da vítima 
[Ver escala](https://pt.wikipedia.org/wiki/Escala_de_coma_de_Glasgow#/media/Ficheiro:Escala_de_Coma_de_Glasgow_-_ECG.png)
</small>

---

## Probabilidade de Sobrevivência


| Classificação START | Faixa de probabilidade de sobrevivência |
|---------------------|-----------------------------------------|
| 0: Verde               | [0.93 – 1.00]                           |
| 1: Amarelo             | [0.75 – 0.95]                           |
| 2: Vermelho            | [0.15 – 0.80]                           |
| 3: Preto               | [0.00 – 0.25]                           |


---

