from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Definir o caminho do diretório para salvar os gráficos
diretorio_graficos = r"C:\Users\Nayane Jacyara\Documents\Faculdade\sistemaInteligentes\Redes_Neurais\Rede_Neural_Multilayer_Perceptron\graficos"
os.makedirs(diretorio_graficos, exist_ok=True)

# Carregar o arquivo .arff
dados, meta = arff.loadarff(r"C:\Users\Nayane Jacyara\Documents\Faculdade\sistemaInteligentes\Redes_Neurais\Rede_Neural_Multilayer_Perceptron\diabetes.arff")

# Converter para DataFrame
df = pd.DataFrame(dados)

# Verificar os nomes das colunas
print("Nomes das colunas:", df.columns)

# Decodificar valores binários para strings (se necessário)
if 'class' in df.columns:
    df['class'] = df['class'].apply(lambda x: x.decode('utf-8'))
elif 'classe' in df.columns:
    df['classe'] = df['classe'].apply(lambda x: x.decode('utf-8'))
else:
    raise KeyError("Coluna 'class' ou 'classe' não encontrada no DataFrame.")

# Converter as classes para valores numéricos
if 'class' in df.columns:
    df['class'] = df['class'].map({'tested_positive': 1, 'tested_negative': 0})
elif 'classe' in df.columns:
    df['classe'] = df['classe'].map({'tested_positive': 1, 'tested_negative': 0})

# Separar features (X) e target (y)
X = df.drop('class', axis=1, errors='ignore')  # Usar 'errors=ignore' para evitar erro caso a coluna não exista
y = df.get('class', df.get('classe'))

# Normalizar os dados
normalizador = StandardScaler()
X = normalizador.fit_transform(X)

# Definir os valores das taxas de aprendizado e número de neurônios na camada escondida
taxas_aprendizado = [0.1, 0.01, 0.001]
tamanhos_camadas_ocultas = [(3), (5), (7)]

# Definir a quantidade de execuções
n_execucoes = 30

# Armazenar os resultados
resultados = []

# Executar o experimento n_execucoes vezes
for i in range(n_execucoes):
    print(f"Execução {i+1}/{n_execucoes}")
    
    # Dividir em conjunto de treino e teste (75% treino, 25% teste)
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, stratify=y, random_state=i)
    
    # Treinar e avaliar cada combinação de taxa de aprendizado e tamanho da camada oculta
    for taxa in taxas_aprendizado:
        for tamanho_camada_oculta in tamanhos_camadas_ocultas:
            # Inicializar e treinar o modelo MLP
            mlp = MLPClassifier(hidden_layer_sizes=tamanho_camada_oculta, learning_rate_init=taxa, max_iter=1000, random_state=i)
            mlp.fit(X_treino, y_treino)

            # Fazer previsões
            y_predito = mlp.predict(X_teste)

            # Calcular o erro médio quadrático (MSE)
            erro_quadratico_medio = mean_squared_error(y_teste, y_predito)

            # Gerar a matriz de confusão
            matriz_confusao = confusion_matrix(y_teste, y_predito)

            # Armazenar os resultados
            resultados.append({
                'iteracao': i + 1,
                'taxa_aprendizado': taxa,
                'tamanho_camadas_ocultas': tamanho_camada_oculta,
                'erro_quadratico_medio': erro_quadratico_medio,
                'matriz_confusao': matriz_confusao.tolist()  # Converter para lista para salvar no CSV
            })

# Converter os resultados para DataFrame para análise
resultados_df = pd.DataFrame(resultados)

# Adicionar um agrupamento para análise estatística
estatisticas_resumo = resultados_df.groupby(['taxa_aprendizado', 'tamanho_camadas_ocultas']).agg(
    mse_media=('erro_quadratico_medio', 'mean'),
    mse_desvio_padrao=('erro_quadratico_medio', 'std'),
    matriz_confusao_media=('matriz_confusao', lambda x: np.mean(np.array(x.tolist()), axis=0))
).reset_index()

print("\nResumo Estatístico Detalhado:")
for index, row in estatisticas_resumo.iterrows():
    taxa_aprendizado = row['taxa_aprendizado']
    tamanho_camadas_ocultas = row['tamanho_camadas_ocultas']
    mse_media = row['mse_media']
    mse_desvio_padrao = row['mse_desvio_padrao']
    matriz_confusao_media = row['matriz_confusao_media']

    print(f"\nTaxa de Aprendizado: {taxa_aprendizado}")
    print(f"Tamanho da Camada Oculta: {tamanho_camadas_ocultas}")
    print(f"Erro Quadrático Médio (MSE) - Média: {mse_media:.4f}, Desvio Padrão: {mse_desvio_padrao:.4f}")
    
    print("Matriz de Confusão Média:")
    print(f"[[{int(matriz_confusao_media[0, 0])}, {int(matriz_confusao_media[0, 1])}]")
    print(f" [ {int(matriz_confusao_media[1, 0])}, {int(matriz_confusao_media[1, 1])}]]")

# Plotar o erro médio quadrático (MSE) para cada combinação de taxa de aprendizado e tamanho da camada oculta
plt.figure(figsize=(14, 7))
sns.lineplot(data=resultados_df, x='iteracao', y='erro_quadratico_medio', hue='taxa_aprendizado', style='tamanho_camadas_ocultas', markers=True)
plt.title('Erro Quadrático Médio (MSE) por Iteração')
plt.xlabel('Iteração')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.legend(title='Taxa de Aprendizado e Tamanho da Camada Oculta')
plt.grid(True)
plt.savefig(os.path.join(diretorio_graficos, 'mse_por_iteracao.png'))
plt.close()

# Plotar a matriz de confusão média para cada combinação de taxa de aprendizado e tamanho da camada oculta
for taxa in taxas_aprendizado:
    for tamanho in tamanhos_camadas_ocultas:
        matriz_media = estatisticas_resumo[
            (estatisticas_resumo['taxa_aprendizado'] == taxa) &
            (estatisticas_resumo['tamanho_camadas_ocultas'] == tamanho)
        ]['matriz_confusao_media'].values[0]

        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_media, annot=True, fmt='g', cmap='Blues', cbar=False, 
                    xticklabels=['Classe Previu Negativa', 'Classe Previu Positiva'],
                    yticklabels=['Classe Real Negativa', 'Classe Real Positiva'])
        plt.title(f'Matriz de Confusão Média - Taxa de Aprendizado: {taxa}, Tamanho da Camada Oculta: {tamanho}')
        plt.xlabel('Classe Previu')
        plt.ylabel('Classe Real')
        plt.savefig(os.path.join(diretorio_graficos, f'matriz_confusao_taxa_{taxa}_tamanho_{tamanho}.png'))
        plt.close()

# Caminho do arquivo CSV para salvar os resultados
caminho_arquivo = r"C:\Users\Nayane Jacyara\Documents\Faculdade\sistemaInteligentes\Redes_Neurais\Rede_Neural_Multilayer_Perceptron\resultados_experimentos.csv"

# Salvar o arquivo CSV no caminho especificado
resultados_df.to_csv(caminho_arquivo, index=False, quoting=1)
