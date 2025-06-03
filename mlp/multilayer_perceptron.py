from neuron import Neuron

class MultilayerPerceptron:
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

    def __init__(self, topology: list, population_size: int = 100):
        """
        Inicializa a MLP com a topologia especificada.

        Cada neurônio em cada camada (exceto a de entrada) é instanciado com a quantidade de entradas definida pela topologia.
        """
        self._neurons = [[Neuron(topology[0]) for _ in range(topology[n])] for n in range(len(topology))]
        self._pop_size = population_size
        self._topology = topology

    def count_weights(self) -> int:
        """
        Retorna o número total de pesos (incluindo bias) necessários para a rede.
        """
        return sum((neuron.n + 1) for layer in self._neurons for neuron in layer)

    def get_neurons_quantity(self, layers) -> int:
        """
        Retorna o número total de neurônios em uma lista de camadas.
        """
        n = 0
        for layer in layers:
            n += len(layer)
        return n

    def get_layers(self) -> list:
        """
        Retorna as camadas ocultas e de saída da MLP (ignora a camada de entrada).
        """
        return self._neurons[1:]

    def predict(self, board: list) -> int:
        """
        Realiza a propagação para frente na MLP usando a entrada fornecida.

        Retorna:
        --------
        int : índice da saída com maior ativação (posição de maior valor no vetor final da rede).
        """
        input = board
        for layer in self.get_layers():
            input = [neuron.decide(input) for neuron in layer]
        return input.index(max(input))  # Retorna o índice do maior valor como decisão final.

    def to_json(self) -> dict:
        """
        Serializa a estrutura e os pesos da rede em um dicionário JSON.
        """
        return {
            'topology': self._topology,
            'population_size': self._pop_size,
            'neurons': [neuron.to_json() for neuron in self._neurons]
        }

    def load_weights_from_vector(self, weights_vector: list):
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

    @staticmethod
    def from_json(json: dict) -> 'MultilayerPerceptron':
        """
        Cria uma nova instância da MLP a partir de um dicionário JSON.

        Parâmetros:
        -----------
        json : dict
            Dicionário no formato exportado pelo método to_json().
        """
        mlp = MultilayerPerceptron(json['topology'], json['population_size'])
        mlp.neurons = []
        for layer_json in json['neurons']:
            layer = []
            for neuron_json in layer_json:
                layer.append(Neuron.from_json(neuron_json))
            mlp.neurons.append(layer)
        return mlp
