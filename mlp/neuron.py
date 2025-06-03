import numpy as np
from random import random
class Neuron:
    def __init__(self, n_params:int, randomize:bool=True):
        self.n = n_params
        self.weights =  np.zeros(shape=[1, n_params + 1], dtype=float) # beta, length  =params + 1
        if randomize:
            for i in range(len(weights)):
                weights[i] = random()

    def decide(self, inputs:list) -> float:
        if len(inputs) > self.n:
            raise ValueError
        # FIXME: Should return probabilty value
        z = np.dot(self.weights, inputs)
        return self.activation_function(z)

    # TODO: activation function (?)
    def activation_function(self, x) -> float:
        return 1 / (1 + np.exp(-x))

    def adjust_weights(self, fitted_weights:list):
        if len(fitted_weights) > len(self.weights):
            raise ValueError
        self.weights = fitted_weights

    def to_json(self):
        return {
                    'n_params' : self.n_params,
                    'weights' : self.weights
                }

    def from_json(self,