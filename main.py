import json
from genetic_algorithm import GeneticAlgorithm
from genetic_algorithm import FitnessEvaluator

# Models
from model import MultilayerPerceptron
from model import Minimax


def train(learner: MultilayerPerceptron, trainer: Minimax, population_size:int, pipeline=list[str], max_iter:int=100, threshold:float=0.95, learning_rate:float=0.1, verbose:bool=False) -> MultilayerPerceptron:
    print(f"Partidas : Easy={pipeline.count('easy')} : Medium={pipeline.count('medium')} : Hard={pipeline.count('hard')}")
    evaluator = FitnessEvaluator(learner, trainer, pipeline)

    training = GeneticAlgorithm(
        pop_size         =population_size,
        chromosome_size  =learner.count_weights(),
        fitness_function =evaluator,
        max_iter         =max_iter,
        learning_rate    =learning_rate
    )

    training.run(verbose=verbose, threshold=threshold)

if __name__ == '__main__':
    PIPELINE = (
    3 * ['medium'] +
    5 * ['hard']
    )

    TOPOLOGY=[9, 9, 9]

    VERBOSE = True

    model = MultilayerPerceptron(TOPOLOGY)
    minimax = Minimax()

    train(
        model,
        minimax,
        population_size=10000,
        pipeline=PIPELINE,
        max_iter=4000,
        learning_rate=0.5,
        verbose=VERBOSE,
        )

    with open('output/model.json', 'w') as f:
        f.write(str(model.to_json()))