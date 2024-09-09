# Projeto: Rede Neural Multilayer Perceptron para Diagnóstico de Diabetes

## Descrição do Projeto

Este projeto utiliza uma Rede Neural Multilayer Perceptron (MLP) para realizar a classificação de dados de pacientes, visando identificar casos de diabetes. O modelo é treinado e testado com diferentes combinações de taxa de aprendizado e número de neurônios na camada escondida, com o objetivo de encontrar a melhor configuração para o diagnóstico.

O dataset utilizado é o **"Diabetes.arff"**, disponível na biblioteca `scipy`, contendo registros de pacientes e seus atributos relacionados à saúde, bem como o diagnóstico de diabetes.

## Estrutura do Projeto

- **diabetes.arff**: O arquivo de dados no formato `.arff` contendo os registros dos pacientes.
- **MLP.py**: O script principal que executa o treinamento e teste da Rede Neural MLP, salvando os resultados e exibindo estatísticas de desempenho.
- **resultados_experimentos.csv**: Arquivo de saída gerado pelo script contendo os resultados detalhados de cada execução.
- **graficos**: Pasta onde os gráficos gerados são salvos.

## Tecnologias Utilizadas

- **Python 3.12**: Linguagem de programação utilizada no projeto.
- **Bibliotecas**:
  - `scipy`: Para carregar os dados no formato `.arff`.
  - `pandas`: Manipulação de dados.
  - `scikit-learn`: Para normalização dos dados, criação do modelo MLP e cálculo de métricas.
  - `numpy`: Cálculos numéricos e manipulação de arrays.
  - `seaborn`: Para criação dos gráficos de matriz de confusão.
  - `matplotlib`: Para salvar os gráficos.

## Como Executar

1. **Pré-requisitos**:
   - Certifique-se de ter o Python 3.12 (ou superior) instalado.
   - Instale as dependências necessárias executando o comando:

   ```bash
   pip install -r requirements.txt
   ```

   Ou, instale as bibliotecas individualmente:

   ```bash
   pip install scipy pandas scikit-learn numpy seaborn matplotlib
   ```

2. **Executar o Script**:
   - Execute o script `MLP.py` para treinar e testar o modelo. O script realizará 30 execuções do treinamento e teste do modelo com diferentes combinações de parâmetros.

   ```bash
   python MLP.py
   ```

3. **Resultados**:
   - O script exibirá um resumo no terminal e salvará os resultados detalhados em um arquivo CSV (`resultados_experimentos.csv`).
   - Gráficos das matrizes de confusão serão salvos na pasta `graficos`.

## Resultados Esperados

O modelo treina e testa a rede neural MLP com as seguintes combinações de parâmetros:

- **Taxas de Aprendizado**: `0.1`, `0.01`, `0.001`
- **Tamanho da Camada Escondida**: `3`, `5`, `7` neurônios

A performance é avaliada com base nas métricas:

- **Erro Médio Quadrático (MSE)**: Quanto menor, melhor.
- **Matriz de Confusão**: Avalia os acertos (verdadeiros positivos/negativos) e erros (falsos positivos/negativos) do modelo.

Os gráficos gerados ajudam na visualização da matriz de confusão média para cada combinação de taxa de aprendizado e tamanho da camada escondida, permitindo uma melhor análise do desempenho do modelo.
