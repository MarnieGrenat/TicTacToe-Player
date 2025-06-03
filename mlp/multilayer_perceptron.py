from genetic_algorithm import Genetic_algorithm
from neuron import Neuron

class mlp:
    def __init__(self, topology:list, population_size:int=100):
        self.neurons = [[Neuron(topology[0]) for _ in range(topology[n])] for n in range(len(topology))]
        self.pop_size = population_size

        self.persist()


    def get_neurons_quantity(self, layers) -> int:
        n = 0
        for layer in layers:
            n = len(layer)
        return n

    def get_layers(self) -> list:
        return self.neurons[1:]

    def predict(self, board:list) -> int:
        input = board
        for layer in self.get_layers():
            input = [neuron.decide(input) for neuron in layer]
        # Return index of max probability
        return input.index(max(input))

    def penalize(self, vector: list):
        n_neurons = self.get_neurons_quantity(self.get_layers())
        for layer in self.get_layers():
            for neuron in layer:
                genetic = Genetic_algorithm(rows=self.pop_size, columns=n_neurons, vector=vector)
                fitness = genetic.fit()
                neuron.adjust_weights(fitness)

    def persist(self):
        # FIXME: Salvar em disco
        pass