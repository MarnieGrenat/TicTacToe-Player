from .model_interface import IModel
from ._neuron import Neuron
import numpy as np
class MultilayerPerceptron(IModel):
    """
    Rede Neural Perceptron Multicamadas (MLP) para problemas de aprendizado supervisionado.

    Esta implementação permite:
    - Criar uma MLP com uma topologia arbitrária.
    - Usar o Algoritmo Genético (AG) para otimizar os pesos da rede.
    - Prever saídas a partir de entradas.
    - Serializar e carregar o modelo via JSON.

    Parâmetros:
    -----------
    topology : list
        Lista de inteiros definindo a quantidade de neurônios em cada camada da MLP.
        Exemplo: [9, 9, 9] cria uma rede com 9 neurônios de entrada, 9 na camada oculta, 9 na saída.
    population_size : int, default=100
        Tamanho da população usada no AG para treinar os pesos (não utilizado diretamente na MLP, mas armazenado para referência).

    Métodos:
    --------
    count_weights() -> int
        Retorna o número total de pesos (incluindo bias) necessários para a rede.

    get_layers() -> list
        Retorna as camadas ocultas e de saída da MLP (ignora a camada de entrada).

    predict(board: list) -> int
        Realiza a propagação para frente na MLP e retorna a posição de maior ativação (índice do maior valor).

    to_json() -> dict
        Serializa a estrutura e pesos da rede em um dicionário JSON.

    load_weights_from_vector(weights_vector: list)
        Aplica um vetor linear de pesos em toda a rede.

    from_json(json: dict) -> MultilayerPerceptron
        Cria uma instância da MLP a partir de um dicionário JSON.
    """

    '''
    TODO LIST:
    -   Arrumar a primeira layer pra sempre ser o tabuleiro(?)
    -   Corrigir a matriz de neurônios para não necessitar ser quadrada
    -
    '''
    def __init__(self, topology: list):
        """
        # Setta neurônio da primeira layer pra frente, pois a primeira layer deve ser o tabuleiro
        Inicializa a MLP com a topologia especificada.

        Cada neurônio em cada camada (exceto a de entrada) é instanciado com a quantidade de entradas definida pela topologia.
        """
        self._topology = topology
        self.set_verbose(False)

        self._neurons = []

        for i in range(1, len(topology)):
            n_inputs = topology[i - 1] # Número de neurônios da camada anterior
            n_outputs = topology[i]     # Número de neurônios da camada atual
            layer = [Neuron(n_params=n_inputs) for _ in range(n_outputs)]
            self._neurons.append(layer)

    def set_verbose(self, verbose:bool) -> None:
        self._verbose = verbose

    def count_weights(self) -> int:
        """
        Retorna o número total de pesos (incluindo bias) necessários para a rede.
        """
        return sum((neuron.n + 1) for layer in self._neurons for neuron in layer)

    def predict(self, board: list[int]) -> int:
        """
        Realiza a propagação para frente na MLP usando a entrada fornecida.

        Retorna:
        --------
        int : índice da saída com maior ativação (posição de maior valor no vetor final da rede).
        """
        input = board
        for layer in self._neurons:
            input = [neuron.decide(input) for neuron in layer]
        output = self._softmax(input)
        if self._verbose:
            print(f'\nMultilayerPerceptron : {output}')
        return int(np.argmax(output))  # Retorna o índice do maior valor como decisão final.

    def _softmax(self, x):
        # sso faz com que a rede normalize os outputs da camada final em probabilidades bem distribuídas,
        # forçando a rede a escolher uma célula de forma mais assertiva.
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def update(self, weights_vector: list) -> None:
        """
        Aplica um vetor linear de pesos em toda a rede.

        O vetor de pesos deve conter todos os pesos e bias da rede concatenados em uma única lista.
        """
        idx = 0
        for layer in self._neurons:
            for neuron in layer:
                n_params = neuron.n + 1
                neuron.adjust_weights(weights_vector[idx:idx + n_params])
                idx += n_params

    def to_json(self) -> dict:
        """
        Serializa a estrutura e os pesos da rede em um dicionário JSON.
        """
        all_neurons = []
        for layer in self._neurons:
            layer_neurons = []
            for neuron in layer:
                layer_neurons.append(neuron.to_json())
            all_neurons.append(layer_neurons)

        return {
            "topology": self._topology,
            "neurons": all_neurons
        }

    @staticmethod
    def from_json(json: dict) -> 'MultilayerPerceptron':
        """
        Cria uma nova instância da MLP a partir de um dicionário JSON.

        Parâmetros:
        -----------
        json : dict
            Dicionário no formato exportado pelo método to_json().
        """
        mlp = MultilayerPerceptron(list(json['topology']))
        mlp._neurons = []
        for layer_json in list[dict](json['neurons']):
            layer = []
            for neuron_json in layer_json:
                layer.append(Neuron.from_json(neuron_json))
            mlp._neurons.append(layer)
        return mlp
