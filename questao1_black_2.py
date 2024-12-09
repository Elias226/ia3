import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados do arquivo spiral.csv
data = np.loadtxt('spiral.csv', delimiter=',')
X = data[:, :2]  # Features (p = 2)
y = data[:, 2]   # Labels (C = 2)

# Visualização inicial dos dados
plt.figure(figsize=(8, 6))
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color="red", label="Classe -1", alpha=0.7)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="Classe 1", alpha=0.7)
plt.title("Visualização Inicial dos Dados - Gráfico de Dispersão")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Função para normalizar os dados
def normalize_data(X):
    """
    Normaliza os dados para que cada feature tenha média 0 e desvio padrão 1.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# Função para dividir os dados em treino e teste
def split_train_test(X, y, train_size=0.8, seed=12):
    """
    Divide os dados em conjuntos de treino e teste.
    Args:
    - X: matriz de features.
    - y: vetor de rótulos.
    - train_size: proporção dos dados para treinamento (default 80%).
    - seed: semente para replicabilidade.
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    # np.random.seed(seed)
    # indices = np.arange(len(X))
    # np.random.shuffle(indices)
    
    indices = np.random.permutation(len(X))

    train_count = int(train_size * len(X))
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Classe Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None

    def signal(self, value):
        """Função de ativação que aplica o sinal."""
        return 1 if value >= 0 else -1

    def fit(self, X, y):
        """
        Treina o Perceptron de acordo com o pseudocódigo fornecido.
        """
        N, p = X.shape
        self.weights = np.zeros(p)  # Inicializa os pesos com 0
        erro_existe = True
        t = 0

        while erro_existe and t < self.max_epochs:
            erro_existe = False
            for i in range(N):
                u = np.dot(self.weights, X[i])  # Calcula u(t)
                y_pred = self.signal(u)        # Calcula y(t)
                # Atualiza os pesos caso d(t) != y(t)
                if y[i] != y_pred:
                    erro_existe = True
                    self.weights += self.learning_rate * (y[i] - y_pred) * X[i]
            t += 1

    def predict(self, X):
        """
        Realiza predições para o conjunto de dados X.
        """
        predictions = []
        for x in X:
            u = np.dot(self.weights, x)  # Calcula u(t)
            y_pred = self.signal(u)      # Calcula y(t)
            predictions.append(y_pred)
        # print(np.array(predictions))
        return np.array(predictions)

class Adaline:
    def __init__(self, learning_rate=0.01, max_epochs=100, epsilon=1e-3):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.w = None


    def signal(self, value):
        """Função de ativação que aplica o sinal."""
        return 1 if value >= 0 else -1
    
    def fit(self, X, d):
        # X: matriz de entrada (N amostras, M atributos)
        # d: vetor de saídas desejadas (N amostras)
        N, M = X.shape
        # Inicialização do vetor de pesos
        self.w = np.zeros(M)
        epoch = 0

        while True:
            EQM_anterior = self._eqm(X, d)
            for t in range(N):
                u = np.dot(self.w, X[t])
                self.w = self.w + self.learning_rate * (d[t] - u) * X[t]
            epoch += 1
            EQM_atual = self._eqm(X, d)
            print(EQM_atual)
            if abs(EQM_atual - EQM_anterior) <= self.epsilon:
                break
            if epoch >= self.max_epochs:
                break
            EQM_anterior = EQM_atual

    def _eqm(self, X, d):
        N = X.shape[0]
        u = X.dot(self.w)
        y_hat = np.array([self.signal(u[i]) for i in range(N)])
        erro = d - y_hat
        EQM = np.sum(erro*2) / (2*N)
        return EQM

    def predict(self, X_teste):
        # u = np.dot(self.w, x)
        # y = -1 if u < 0 else 1
        # return y

        predictions = []
        for x in X_teste:
            u = np.dot(self.w, x)  # Calcula u(t)
            y_pred = self.signal(u)      # Calcula y(t)
            predictions.append(y_pred)
        # print(np.array(predictions))
        return np.array(predictions)
    

class MLP:
    def __init__(self, L, qtd_neuronios, m, eta, maxEpoch, criterio_parada):
        # Lógica inicial é mantida igual
        self.L = L
        self.qtd_neuronios = qtd_neuronios
        self.m = m
        self.eta = eta
        self.maxEpoch = maxEpoch
        self.criterio_parada = criterio_parada
        self.W = []

    def inicializar_pesos(self):
        self.W = []
        for i in range(len(self.arquitetura)-1):
            W_i = np.random.uniform(-0.5, 0.5, (self.arquitetura[i+1], self.arquitetura[i]))
            self.W.append(W_i)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, y):
        return y * (1.0 - y)

    def forward(self, x_amostra):
        y = []
        y_interm = x_amostra.copy()
        y.append(y_interm)
        
        for j in range(len(self.W)):
            i_interm = np.dot(self.W[j], y_interm)
            y_interm = self.sigmoid(i_interm)
            if j < len(self.W)-1:
                y_interm = np.vstack((-1*np.ones((1,1)), y_interm))
            y.append(y_interm)
        return y

    def backward(self, y, d):
        delta = []
        L = len(self.W) - 1
        delta_L = (d - y[-1]) * self.sigmoid_derivative(y[-1])
        delta.insert(0, delta_L)
        
        for j in range(L, 0, -1):
            W_prox = self.W[j]
            delta_prox = delta[0]
            y_sem_bias = y[j]

            if j != L:
                deriv = self.sigmoid_derivative(y_sem_bias[1:, :])
                W_prox_sem_bias = W_prox[:, 1:]
                delta_j = deriv * np.dot(W_prox_sem_bias.T, delta_prox)
            else:
                deriv = self.sigmoid_derivative(y_sem_bias[1:, :])
                W_prox_sem_bias = W_prox
                delta_j = deriv * np.dot(W_prox_sem_bias.T, delta_prox)
            delta.insert(0, delta_j)
        
        for j in range(len(self.W)):
            self.W[j] += self.eta * np.dot(delta[j], y[j].T)

    def calcular_eqm(self, X_treino, Y_treino):
        N = X_treino.shape[1]
        EQM = 0.0
        for n in range(N):
            x_amostra = X_treino[:, n].reshape(-1, 1)
            d = Y_treino[:, n].reshape(-1, 1)
            y = self.forward(x_amostra)
            EQM += np.sum((d - y[-1])**2)
        return EQM / (2 * N)

    def fit(self, X_treino, Y_treino):
        """
        Treinamento do MLP.
        X_treino: matriz de características (N amostras, p atributos).
        Y_treino: rótulos (N amostras, 1 ou mais classes).
        """
        N, p = X_treino.shape
        # self.X_treino = np.c_[-np.ones(N), X_treino]  # Adiciona coluna de bias
        self.X_treino = X_treino
        self.Y_treino = Y_treino

        self.arquitetura = [self.X_treino.shape[1]] + self.qtd_neuronios + [self.m]
        self.inicializar_pesos()

        epoch = 0
        EQM = 1e10

        while (EQM > self.criterio_parada) and (epoch < self.maxEpoch):
            for n in range(N):
                x_amostra = self.X_treino[n].reshape(-1, 1)  # Seleciona a linha n como coluna
                d = self.Y_treino[n].reshape(-1, 1)  # Seleciona o rótulo n como coluna
                y = self.forward(x_amostra)
                self.backward(y, d)
            EQM = self.calcular_eqm(self.X_treino.T, self.Y_treino.T)
            epoch += 1


    def predict(self, X_teste):
        X_teste_bias = np.vstack((-1 * np.ones((1, X_teste.shape[1])), X_teste))
        saidas = []
        for n in range(X_teste_bias.shape[1]):
            x_amostra = X_teste_bias[:, n].reshape(-1, 1)
            y = self.forward(x_amostra)
            saidas.append(np.argmax(y[-1]))
        return saidas


# Função para calcular a matriz de confusão
def confusion_matrix(y_true, y_pred):
    """
    Calcula a matriz de confusão para dois rótulos (-1 e 1).
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
    TN = np.sum((y_true == -1) & (y_pred == -1))  # True Negative
    FP = np.sum((y_true == -1) & (y_pred == 1))  # False Positive
    FN = np.sum((y_true == 1) & (y_pred == -1))  # False Negative
    return np.array([[TP, FP], [FN, TN]])

# Função para validação Monte Carlo
def monte_carlo_validation_with_confusion(model, X, y, num_iterations=500, train_size=0.8, seed=12):
    """
    Validação Monte Carlo com cálculo da matriz de confusão.
    """
    acuracias = []
    sensibilidades = []
    especificidades = []
    matrizes_confusao = []

    for i in range(num_iterations):

        if (i + 1) % 50 == 0:
            print(f"Rodada {i + 1}/{num_iterations}: Acurácia média até agora = {np.mean(acuracias):.4f}")
        # print(f"Rodada {i + 1}/{num_iterations}: Acurácia média até agora = {np.mean(acuracias):.4f}")
        X_train, X_test, y_train, y_test = split_train_test(X, y, train_size=train_size, seed=seed + i)

        # Treinar o modelo
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_test)

        # Calcular a matriz de confusão para a rodada atual
        matriz_confusao = confusion_matrix(y_test, y_pred)
        matrizes_confusao.append(matriz_confusao)

        # Calcular métricas
        TP, FN = matriz_confusao[0]
        FP, TN = matriz_confusao[1]
        acuracias.append((TP + TN) / (TP + TN + FP + FN))
        sensibilidades.append(TP / (TP + FN) if TP + FN > 0 else 0)
        especificidades.append(TN / (TN + FP) if TN + FP > 0 else 0)

    # Identificar melhor e pior acurácia
    melhor_idx = np.argmax(acuracias)
    pior_idx = np.argmin(acuracias)
    
    resultados = {
        "acuracia_media": np.mean(acuracias),
        "acuracia_desvio_padrao": np.std(acuracias),
        "acuracia_maxima": np.max(acuracias),
        "acuracia_minima": np.min(acuracias),

        "sensibilidade_media": np.mean(sensibilidades),
        "sensibilidade_desvio_padrao": np.std(sensibilidades),
        "sensibilidade_maxima": np.max(sensibilidades),
        "sensibilidade_minima": np.min(sensibilidades),

        "especificidade_media": np.mean(especificidades),
        "especificidade_desvio_padrao": np.std(especificidades),
        "especificidade_maxima": np.max(especificidades),
        "especificidade_minima": np.min(especificidades),

        "melhor_matriz": matrizes_confusao[melhor_idx],
        "pior_matriz": matrizes_confusao[pior_idx]
    }
    return resultados

def plot_matriz_confusao(matriz, titulo, acuracia, color):
    """
    Plota uma matriz de confusão estilizada no formato do exemplo.
    
    Args:
        matriz: Matriz de confusão (numpy array 2x2).
        titulo: Título do gráfico.
        acuracia: Valor da acurácia.
    """
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(matriz, annot=True, fmt="d", cmap=color , cbar=False, linewidths=1.5, linecolor="black")
    
    # Ajustar o título com as métricas
    plt.title(f"{titulo}\nAcurácia: {acuracia:.2f}",
              fontsize=12, pad=20)
    
    # Configurar os rótulos dos eixos
    plt.xlabel("Real", fontsize=12)
    plt.ylabel("Predito", fontsize=12)
    plt.xticks([0.5, 1.5], ["+1", "-1"], fontsize=10)
    plt.yticks([0.5, 1.5], ["+1", "-1"], fontsize=10)
    
    plt.show()


# Plot da reta de decisão sobre o gráfico de dispersão
def plot_decision_boundary(model, X, y):
    """
    Plota a reta de decisão do modelo Perceptron.
    """
    # Coeficientes da reta (pesos)
    w = model.weights
    if w[2] == 0:  # Evitar divisão por zero
        print("A reta de decisão não pode ser calculada (w[2] = 0).")
        return
    
    # Calcula os limites do eixo X
    x_min, x_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # Calcula os valores correspondentes de x2 (feature no eixo Y)
    x1_range = np.linspace(x_min, x_max, 100)
    x2_range = -(w[0] + w[1] * x1_range) / w[2]

    # Plot dos dados
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == -1][:, 1], X[y == -1][:, 2], color="red", label="Classe -1", alpha=0.7)
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], color="blue", label="Classe 1", alpha=0.7)
    # Plot da reta de decisão
    plt.plot(x1_range, x2_range, color="green", label="Reta de Decisão", linewidth=2)

    # Configurações do gráfico
    plt.title("Reta de Decisão do Perceptron")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def printar(resultados):
    # Exibir os resultados
    print("Resultados da Validação Monte Carlo:")
    print(f"\nAcurácia Média: {resultados['acuracia_media']:.4f}")
    print(f"Acurácia Desvio Padrão: {resultados['acuracia_desvio_padrao']:.4f}")
    print(f"Acurácia Máxima: {resultados['acuracia_maxima']:.4f}")
    print(f"Acurácia Mínima: {resultados['acuracia_minima']:.4f}")

    print(f"\nSensibilidade Média: {resultados['sensibilidade_media']:.4f}")
    print(f"Sensibilidade Desvio Padrão: {resultados['sensibilidade_desvio_padrao']:.4f}")
    print(f"Sensibilidade Máxima: {resultados['sensibilidade_maxima']:.4f}")
    print(f"Sensibilidade Mínima: {resultados['sensibilidade_minima']:.4f}")

    print(f"\nEspecificidade Média: {resultados['especificidade_media']:.4f}")
    print(f"Especificidade Desvio Padrão: {resultados['especificidade_desvio_padrao']:.4f}")
    print(f"Especificidade Máxima: {resultados['especificidade_maxima']:.4f}")
    print(f"Especificidade Mínima: {resultados['especificidade_minima']:.4f}")


# Normalizar os dados e adicionar a coluna de bias
# X = normalize_data(X)
N, p = X.shape
X = np.c_[-np.ones(N), X]  # Adiciona coluna de bias (-1)

# # Criar o modelo e realizar a validação
# perceptron = Perceptron(learning_rate=0.1, max_epochs=100)
# perceptron.fit(X, y)
# # plot_decision_boundary(perceptron, X, y)

# resultados_perceptron = monte_carlo_validation_with_confusion(perceptron, X, y, num_iterations=500, train_size=0.8)

# # Plotar a melhor matriz de confusão
# plot_matriz_confusao(resultados_perceptron["melhor_matriz"], "Perceptron Simples (Melhor Acurácia)", resultados_perceptron['acuracia_maxima'], "Greens")

# # Plotar a pior matriz de confusão
# plot_matriz_confusao(resultados_perceptron["pior_matriz"], "Perceptron Simples (Pior Acurácia)", resultados_perceptron['acuracia_minima'], "Reds")

# printar(resultados_perceptron)

adaline = Adaline(learning_rate=0.01, max_epochs=100, epsilon=1e-5)

resultados_adaline = monte_carlo_validation_with_confusion(adaline, X, y, num_iterations=500, train_size=0.8)

# Plotar a melhor matriz de confusão
plot_matriz_confusao(resultados_adaline["melhor_matriz"], "ADALINE (Melhor Acurácia)", resultados_adaline['acuracia_maxima'], "Greens")

# Plotar a pior matriz de confusão
plot_matriz_confusao(resultados_adaline["pior_matriz"], "ADALINE (Pior Acurácia)", resultados_adaline['acuracia_minima'], "Reds")

printar(resultados_adaline)

# Exemplo de uso
# model = MLP(L=2, qtd_neuronios=[10, 10], m=2, eta=0.1, maxEpoch=100, criterio_parada=1e-3)
# resultados = monte_carlo_validation_with_confusion(model, X, y, num_iterations=5, train_size=0.8, seed=42)

# print("Resultados:")
# print("Acurácia Média:", resultados["acuracia_media"])
# print("Sensibilidade Média:", resultados["sensibilidade_media"])
# print("Especificidade Média:", resultados["especificidade_media"])
