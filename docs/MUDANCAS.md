# Documentação de Mudanças

Este documento descreve as modificações feitas para corrigir problemas de
cálculo de métricas no pipeline NeuS aplicado ao dataset SonarCloud.

---

## 1. Contexto do Problema

Ao inspecionar o arquivo `experiments/sphere_data_no_terrain_moved_terrain0/0/metrics/00001536.json`,
observamos valores de métricas de geometria **absurdos**:

```json
"geometry": {
  "accuracy": 8.6958,      // metros
  "completeness": 9.0641,  // metros
  "chamfer_l1": 17.7599,   // metros
}
```

E todas as métricas `precision/recall/f1` eram **0.0**, mesmo após 1536
iterações de treinamento. O modelo parecia não estar aprendendo — mas,
na verdade, **as métricas estavam sendo calculadas em espaços de
coordenadas diferentes**, com um offset global de ~8.7 m entre a
malha predita e a nuvem GT.

---

## 2. Diagnóstico

### 2.1. Sistemas de coordenadas incompatíveis

| Origem | Centro | Espaço |
|---------|--------|--------|
| `run_sdf.py` (mesh extraída) | `~[0.37, 0.03, -0.13]` | Bounding box centrado na origem (`[-1.5, 1.5]³`) |
| `reconstruct_data.py` (GT) | `~[-1.08, 5.56, -7.99]` | Coordenadas globais do simulador Unreal Engine |

Distância entre as duas origens:
$$\sqrt{0.37^2 + 0.03^2 + (-0.13)^2} \approx 8.7 \text{ m}$$

Como o KDTree comparava pontos diretamente em espaços não alinhados,
**todos os pares estavam a mais de 1 cm, 2 cm e 5 cm de distância**,
resultando em Precision = Recall = F1 = 0.

### 2.2. Thresholds muito restritivos

O `.conf` original tinha apenas `[0.01, 0.02, 0.05]` (1, 2 e 5 cm).
Esses thresholds só fazem sentido quando o modelo já está bem
refinado. Durante a fase inicial do treino, é normal que a maior parte
dos pontos esteja mais distante.

### 2.3. Confirmação experimental

Testes de alinhamento em `00001536.json` (após iteração 1536):

| Métrica | Sem Alinhamento | Centróide | ICP |
|---------|-----------------|-----------|-----|
| Accuracy | 8.69 m | 0.23 m | 0.20 m |
| Completeness | 9.05 m | 0.65 m | 0.67 m |
| Chamfer L1 | 17.75 m | 0.88 m | 0.87 m |
| F1 @ 2 cm | 0.00% | 3.12% | 3.36% |
| F1 @ 5 cm | 0.00% | 10.75% | 9.06% |
| F1 @ 10 cm | 0.00% | 19.66% | 16.78% |
| F1 @ 20 cm | 0.00% | 27.67% | 27.16% |

**Conclusão**: o modelo estava aprendendo — apenas as métricas estavam
sendo calculadas erradas.

---

## 3. Modificações Aplicadas

### 3.1. `confs/sonar_cloud/template_sonarcloud.conf`

**O que mudou**:
- Adicionada a chave `align = "centroid"` no bloco `metrics`
- Thresholds ampliados de `[0.01, 0.02, 0.05]` para `[0.01, 0.02, 0.05, 0.10, 0.20, 0.50]`

**Por quê**:
- `align = "centroid"` instrui `evaluate_mesh_metrics` a centrar as
  duas nuvens antes de calcular distâncias.
- Thresholds adicionais (10 cm, 20 cm, 50 cm) capturam a evolução do
  treino desde fases iniciais, onde os erros são maiores.

### 3.2. `scripts/recompute_metrics.py` (novo)

**O que faz**:
- Re-calcula as métricas para **todas** as 11 meshes já salvas
  (`00000000.obj` a `00001920.obj`)
- Aplica alinhamento de centróides
- Regenera `metrics/*.json` e `metrics.csv`
- Usa thresholds `[0.01, 0.02, 0.05, 0.10, 0.20]`

**Por que foi necessário**:
- O `.conf` antigo já foi usado para gerar os arquivos `metrics/*.json`
  existentes com valores errados. Reescrever manualmente seria
  tedioso e propenso a erros.
- Este script permite reprocessar qualquer experimento existente sem
  precisar retreinar o modelo.

### 3.3. `scripts/update_best_metrics.py` (novo)

**O que faz**:
- Identifica a melhor mesh segundo o mesmo critério do
  `update_best_metrics()` original (F1 no threshold do meio).
- Reescreve `best_metrics.json` apontando para a iteração vencedora.

**Por quê**:
- O `best_metrics.json` antigo apontava para iteração 0 com F1 = 0.0,
  resultado da falha original no cálculo.

### 3.4. `Train_SonarCloud.sh` (reescrito)

**Problemas do script original**:
1. Template path errado: `confs/template_sonarcloud.conf` (inexistente)
2. Glob superficial `*_data` pegava apenas o nível do objeto, mas o
   dataset real é `*_data/*/moved_terrain*`
3. `python` em vez de `conda run -n aoneus_5090 python` → ModuleNotFoundError
4. Sem `mkdir -p confs/generated`
5. Sem `set -euo pipefail` (erros silenciosos)

**Correções aplicadas**:
- Template path corrigido para `confs/sonar_cloud/template_sonarcloud.conf`
- Glob expandido para `data/SonarCloud/*_data/*/moved_terrain*` (192 cenas)
- Uso de `conda run -n aoneus_5090 python` para garantir o env correto
- `mkdir -p ${CONFS_DIR}` adicionado
- `set -euo pipefail` adicionado
- `expID` único: `${obj}_${terrain}_${moved}` para evitar colisões
- Variáveis configuráveis: `GPU`, `CONDA_ENV`
- Filtro por objeto: `./Train_SonarCloud.sh sphere_data`
- Verificação inicial do template e da lista de cenas

### 3.5. `scripts/dry_run_train.sh` (novo)

**O que faz**:
- Simula o `Train_SonarCloud.sh` gerando os `.conf` mas **sem** rodar
  treino.
- Útil para validar que os paths e placeholders estão corretos antes de
  disparar 192 treinos.

---

## 4. Resultados Após as Correções

### 4.1. Métricas recalculadas (com alinhamento de centróides)

| Iter | Acc ↓ | Comp ↓ | Chamfer L1 ↓ | F1 @ 2 cm | F1 @ 5 cm | F1 @ 20 cm |
|-----:|------:|-------:|-------------:|----------:|----------:|-----------:|
|    0 | 0.301 | 0.728  | 1.029        | 1.42%     | 4.30%     | 18.36%     |
|  768 | 0.246 | 0.773  | 1.020        | 3.39%     | 8.59%     | 21.09%     |
| 1152 | 0.233 | 0.666  | 0.899        | 2.75%     | 9.97%     | 26.69%     |
| 1536 | 0.235 | 0.656  | 0.891        | 2.87%     | 10.32%    | 27.52%     |
| **1920** ⭐ | 0.239 | 0.650 | **0.889** | **5.01%** | **11.99%** | 27.63% |

### 4.2. Melhor iteração

- **`best_metrics.json`** agora aponta para **iteração 1920**
- F1 @ 5 cm = **11.99%**
- Accuracy = 0.239 m
- Chamfer L1 = 0.889 m

### 4.3. Tendência de convergência

Observa-se uma queda consistente no Chamfer L1 de 1.029 m (iter 0)
para 0.889 m (iter 1920), confirmando que o modelo está convergindo.

A Completeness (~0.65 m) é maior que a Accuracy (~0.24 m) porque a
malha tende a subamostrar regiões ocluídas (apenas um lado do objeto
é observado pelo sonar).

---

## 5. Arquivos Modificados / Criados

```
confs/sonar_cloud/template_sonarcloud.conf    [modificado]  align + thresholds
Train_SonarCloud.sh                          [reescrito]   corrigido bugs
scripts/recompute_metrics.py                 [novo]        reprocessa métricas
scripts/update_best_metrics.py               [novo]        atualiza best_metrics
scripts/dry_run_train.sh                     [novo]        simulação sem treino
docs/MUDANCAS.md                             [novo]        este documento
```

Experimentos regerados:
```
experiments/sphere_data_no_terrain_moved_terrain0/0/metrics/*.json  (11)
experiments/sphere_data_no_terrain_moved_terrain0/0/metrics.csv
experiments/sphere_data_no_terrain_moved_terrain0/0/best_metrics.json
```