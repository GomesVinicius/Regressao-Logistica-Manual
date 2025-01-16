# Implementação Manual da Regressão Logística

Este repositório contém uma versão manual do algoritmo de **Regressão Logística**, incluindo a implementação da **Função Sigmoide**.

## Conteúdo do Repositório

- **`logistic_regression_manual.ipynb`**: Implementação manual da Regressão Logística.
- **`Diamonds.ipynb`**: Exemplo prático da Regressão Logística aplicada à categorização de diamantes utilizando a biblioteca **SkLearn**.

## Sobre o Dataset

O dataset original contém três tipos de diamantes. No entanto, o tipo com menor quantidade de dados (**GIA Lab-Grown**) foi excluído. Dessa forma, o modelo foi treinado apenas com os dois tipos restantes:

- **IGI Lab-Grown**
- **GIA**

### Ajustes e Treinamento do Modelo

- O modelo foi treinado com um dataset balanceado contendo **500 registros** de cada classe (**IGI Lab-Grown** e **GIA**).
- Na base de validação, o modelo apresentou ótimos resultados em **precision, recall e F1-score**, mesmo com um conjunto de dados desbalanceado (contendo o dobro de registros de **GIA** em relação a **IGI Lab-Grown**).	

### Expansão para Classificação Multinomial

O objetivo inicial do modelo foi a **classificação binária** (entre IGI Lab-Grown e GIA).
Entretanto, é possível estender o modelo para classificar os três tipos originais de diamantes utilizando a **Regressão Logística Multinomial**.
