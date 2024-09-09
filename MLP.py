from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

# Carregar o arquivo .arff
data, meta = arff.loadarff(r"C:\Users\Nayane Jacyara\Documents\Faculdade\sistemaInteligentes\Redes_Neurais\Rede_Neural_Multilayer_Perceptron\diabetes.arff")

# Converter para DataFrame
df = pd.DataFrame(data)

# Decodificar valores binários para strings
df['class'] = df['class'].apply(lambda x: x.decode('utf-8'))

# Converter as classes para valores numéricos
df['class'] = df['class'].map({'tested_positive': 1, 'tested_negative': 0})

# Separar features (X) e target (y)
X = df.drop('class', axis=1)
y = df['class']

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Definir os valores das taxas de aprendizado e número de neurônios na camada escondida
learning_rates = [0.1, 0.01, 0.001]
hidden_layer_sizes_list = [(3,), (5,), (7,)]

# Armazenar os resultados
results = []

# Executar o experimento 30 vezes
n_runs = 30
for i in range(n_runs):
    print(f"Execução {i+1}/{n_runs}")
    
    # Dividir em conjunto de treino e teste (75% treino, 25% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=i)
    
    for lr in learning_rates:
        for hidden_layer_sizes in hidden_layer_sizes_list:
            # Inicializar e treinar o modelo MLP
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=lr, max_iter=1000, random_state=i)
            mlp.fit(X_train, y_train)

            # Fazer previsões
            y_pred = mlp.predict(X_test)

            # Calcular o erro médio quadrático (MSE)
            mse = mean_squared_error(y_test, y_pred)

            # Gerar a matriz de confusão
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Armazenar os resultados
            results.append({
                'iteration': i + 1,
                'learning_rate': lr,
                'hidden_layer_sizes': hidden_layer_sizes,
                'mse': mse,
                'conf_matrix': conf_matrix.tolist()  # Converter para lista para salvar no CSV
            })

# Converter os resultados para DataFrame para análise
results_df = pd.DataFrame(results)

# Agrupar resultados por taxa de aprendizado e tamanho da camada escondida e calcular médias e desvios padrão
summary_stats = results_df.groupby(['learning_rate', 'hidden_layer_sizes']).agg(
    mse_mean=('mse', 'mean'),
    mse_std=('mse', 'std'),
    conf_matrix_mean=('conf_matrix', lambda x: np.mean(np.array(x.tolist()), axis=0))
).reset_index()

# Imprimir o resumo detalhado
print("\nResumo Estatístico Detalhado:")
for index, row in summary_stats.iterrows():
    lr = row['learning_rate']
    hls = row['hidden_layer_sizes']
    mse_mean = row['mse_mean']
    mse_std = row['mse_std']
    conf_matrix_mean = row['conf_matrix_mean']

    print(f"\nTaxa de Aprendizado: {lr}")
    print(f"Tamanho da Camada Escondida: {hls}")
    print(f"Erro Médio Quadrático (MSE) - Média: {mse_mean:.4f}, Desvio Padrão: {mse_std:.4f}")
    
    print("Matriz de Confusão Média:")
    print(f"[[{int(conf_matrix_mean[0, 0])}, {int(conf_matrix_mean[0, 1])}]")
    print(f" [ {int(conf_matrix_mean[1, 0])}, {int(conf_matrix_mean[1, 1])}]]")

# resultados em um arquivo CSV
results_df.to_csv("resultados_experimentos.csv", index=False, quoting=1)
 