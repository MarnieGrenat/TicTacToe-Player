import numpy as np
from random import random

class Neuron:
    """
    Classe que representa um único neurônio em uma rede neural.

    Cada neurônio possui:
    - Um número de entradas (n_params).
    - Um vetor de pesos, incluindo o bias.
    - Uma função de ativação (sigmoide).
    - Métodos para decidir a saída, ajustar pesos e serializar/deserializar.

    Parâmetros:
    -----------
    n_params : int
        Número de entradas (features) que o neurônio recebe.
    randomize : bool, default=True
        Se True, inicializa os pesos de forma aleatória entre 0 e 1. Caso contrário, inicializa como zeros.

    Métodos:
    --------
    decide(inputs: list) -> float
        Calcula a saída do neurônio a partir das entradas e aplica a função de ativação.

    activation_function(x) -> float
        Função de ativação do neurônio (sigmoide).

    adjust_weights(fitted_weights: list) -> None
        Atualiza os pesos do neurônio a partir de um vetor externo.

    to_json() -> dict
        Serializa o neurônio em formato JSON (dicionário).

    from_json(json: dict) -> Neuron
        Cria uma instância de Neuron a partir de um dicionário JSON.
    """

    def __init__(self, n_params: int, randomize: bool = False):
        """
        Inicializa o neurônio com o número de entradas especificado.
        Por padrão, os pesos (incluindo o bias) são inicializados aleatoriamente.
        """
        self.n = n_params  # Número de entradas
        self.weights = np.zeros(shape=[n_params + 1], dtype=float)  # Pesos + bias

        if randomize:
            for i in range(len(self.weights)):
                self.weights[i] = random()  # Valores entre 0 e 1
        # print(f'Neuron : N={self.n} : Weights={self.weights}')

    def decide(self, inputs: list) -> float:
        """
        Calcula a saída do neurônio a partir das entradas fornecidas.

        Parâmetros:
        -----------
        inputs : list
            Lista de entradas para o neurônio (deve ter tamanho n_params).

        Retorna:
        --------
        float : Valor da saída do neurônio após aplicar a função de ativação.
        """
        if len(inputs) > self.n:
            raise ValueError("Número de entradas maior que o esperado.")

        z = np.dot(self.weights, np.append(inputs, 1))  # Inclui o bias
        return self.activation_function(z)

    def activation_function(self, x) -> float:
        """
        Função de ativação do neurônio (sigmoide).
        """
        return 1 / (1 + np.exp(-x))

    def adjust_weights(self, fitted_weights: list) -> None:
        """
        Ajusta os pesos do neurônio a partir de um vetor externo.

        Parâmetros:
        -----------
        fitted_weights : list
            Lista de novos pesos (incluindo bias). Deve ter o mesmo tamanho que self.weights.
        """
        if len(fitted_weights) != len(self.weights):
            raise ValueError(f"Esperado {len(self.weights)} pesos, recebeu {len(fitted_weights)}")
        self.weights = np.array(fitted_weights)

    def to_json(self) -> dict:
        """
        Serializa o neurônio em um dicionário JSON.

        Retorna:
        --------
        dict : Estrutura contendo o número de entradas e os pesos do neurônio.
        """
        return {
            "n_params": self.n,
            "weights": self.weights.tolist()
        }

    @staticmethod
    def from_json(json: dict) -> 'Neuron':
        """
        Cria uma instância de Neuron a partir de um dicionário JSON.

        Parâmetros:
        -----------
        json : dict
            Dicionário com as chaves 'n_params' e 'weights'.

        Retorna:
        --------
        Neuron : Instância reconstruída.
        """
        n = json['n_params']
        neuron = Neuron(n, randomize=False)
        neuron.weights = np.array(json['weights'])
        return neuron
