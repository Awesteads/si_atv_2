# Conteúdo

Esta pasta contém a definição dos ambientes: os grids com as posições das vítimas. O padrão de nomeação é largura x altura _ número de vítimas.

## env_config.txt
| Constante        | Descrição                                                                 |
|------------------|---------------------------------------------------------------------------|
| BASE 0,0         | posição da base                                                           |
| GRID_WIDTH 12    | largura do grid                                                           |
| GRID_HEIGHT 12   | altura do grid                                                            |
| WINDOW_WIDTH 400 | largura da janela                                                         |
| WINDOW_HEIGHT 400| altura da janela                                                          |
| DELAY 0.0        | delay para animação                                                       |
| STATS_PER_AG 1   | estatísticas são mostradas para cada um dos agentes (1) ou não (0)        |
| STATS_ALL_AG 1   | estatísticas são mostradas de forma acumulada para todos os agentes (1) ou não (0) |

## env_obst.txt
Contém a posição x, y de cada um dos obstáculos com a respectiva dificuldade ]0,100].

## env_victims.txt
Contém a posição das vítimas no grid: x, y





