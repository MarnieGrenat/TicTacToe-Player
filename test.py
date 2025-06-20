import os
import json
from genetic_algorithm import GeneticAlgorithm, FitnessEvaluator
from model import MultilayerPerceptron, Minimax

def load_model(json_path: str) -> MultilayerPerceptron:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return MultilayerPerceptron.from_json(data)

# Salva o modelo

model = load_model('output/model_1.json')
minimax = Minimax()

# Testa o modelo após o treinamento
print("Main : Avaliação contra o Minimax:")
FitnessEvaluator.test_model(model, minimax, rounds=50)
