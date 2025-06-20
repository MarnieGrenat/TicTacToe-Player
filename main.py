import os
import json
from genetic_algorithm import GeneticAlgorithm, FitnessEvaluator
from model import MultilayerPerceptron, Minimax

def train(learner: MultilayerPerceptron, trainer: Minimax, population_size: int, pipeline: list[str],
          max_iter: int = 100, threshold: float = 500, learning_rate: float = 0.1, mutation_rate: float = 0.1,
          verbose: bool = False) -> MultilayerPerceptron:

    print(f"Main : Pipeline de dificuldades: Easy={pipeline.count('easy')} | Medium={pipeline.count('medium')} | Hard={pipeline.count('hard')}")

    training = GeneticAlgorithm(
        pop_size=population_size,
        chromosome_size=learner.count_weights(),
        fitness_function=FitnessEvaluator(learner, trainer, pipeline, verbose),
        max_iter=max_iter,
        learning_rate=learning_rate,
        mutation_rate=mutation_rate,
        verbose=verbose
    )

    best_chromosomes = training.run(threshold=threshold)
    learner.update(best_chromosomes)
    return learner

if __name__ == '__main__':
    # Garante que a pasta existe
    os.makedirs('output', exist_ok=True)

    # Define parâmetros do treinamento
    PIPELINE = (
        5 * ['medium'] +
        8 * ['hard']
    )

    TOPOLOGY = [9, 32, 18, 9]

    VERBOSE = False

    model = MultilayerPerceptron(TOPOLOGY)
    minimax = Minimax()

    # Treina o modelo
    model = train(
        learner=model,
        trainer=minimax,
        population_size=1000,
        pipeline=PIPELINE,
        max_iter=100,
        learning_rate=0.1,
        mutation_rate=0.3,
        threshold=17 * 200, # PipelineLength * MaxEvaluation
        verbose=VERBOSE,
    )

    # Salva o modelo
    with open('output/model_3.json', 'w') as f:
        json.dump(model.to_json(), f)

    # Testa o modelo após o treinamento
#    print("Main : Avaliação contra o Minimax:")
#    FitnessEvaluator.test_model(model, minimax, rounds=50)
