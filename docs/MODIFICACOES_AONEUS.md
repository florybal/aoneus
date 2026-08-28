# Documentação de Modificações - AONeuS para SonarCloud

Este documento detalha as adaptações realizadas no framework AONeuS para viabilizar o uso do dataset SonarCloud.

## Justificativa para o uso do SonarCloud

O dataset SonarCloud é fundamental para o projeto por fornecer:
1. **Dados Sintéticos de Alta Fidelidade:** Essenciais para treinar e validar modelos de reconstrução 3D em ambientes subaquáticos, onde a obtenção de dados reais com *ground truth* preciso é extremamente difícil ou inviável.
2. **Multimodalidade:** Oferece informações complementares (RGB, sonar, profundidade) que permitem contornar as limitações individuais de cada sensor (ex: a turbidez que afeta o RGB é superada pela estrutura do sonar).
3. **Ground Truth:** Fornece poses de câmera e geometria de referência, permitindo uma avaliação quantitativa rigorosa da reconstrução.

---

## Modificações no Código

| Arquivo | Linhas | Descrição da Modificação | Justificativa |
| :--- | :--- | :--- | :--- |
| `NeuS/models/dataset.py` | 110-116 | Patch para tratar máscaras ausentes (`np.ones_like`). | O dataset SonarCloud não fornece máscaras para todas as imagens; o patch evita erros de `FileNotFoundError`. |
| `load_sonarcloud.py` | 218-228 | Patch no `parse_pose_from_path` para parsing robusto de `orientation_X`. | O formato dos nomes de arquivos/pastas continha sufixos que causavam erro no `int()` original. |

---

## Scripts Adicionados

Para viabilizar o pipeline, foram criados os seguintes scripts em `scripts/`:

*   `render_cluster_to_rgb.py`: Renderiza imagens RGB sintéticas a partir de mapas de profundidade.
*   `organize_synthetic_dataset.py`: Estrutura o dataset hierarquicamente (`objeto/terreno/variacao/orientacao/`) e gera `cameras_sphere.npz`.
*   `merge_orientations.py`: Consolida múltiplas orientações em um único diretório, incluindo imagens, profundidade e sonar, para treinamento conjunto.

---

## Configurações

Os arquivos `.conf` em `confs/sonar_cloud/` foram atualizados para:
*   Incluir `general.base_exp_dir`.
*   Apontar corretamente para os novos caminhos de dados organizados.
