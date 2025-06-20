import random
import numpy as np

# Optimization
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

class GeneticAlgorithm:
    """
    Algoritmo Genético para otimização de cromossomos representados como vetores de floats.
    Adaptado para treinamento de redes neurais ou outros problemas de otimização contínua.

    Parâmetros:
    -----------
    pop_size : int
        Tamanho da população.
    chromosome_size : int
        Número de genes (pesos) em cada cromossomo.
    fitness_function : function
        Função de aptidão que recebe um cromossomo (lista de floats) e retorna um valor numérico (fitness).
    max_iter : int, default=100
        Número máximo de gerações para executar o algoritmo.

    Métodos:
    --------
    run(threshold=9.5)
        Executa o ciclo do algoritmo genético até atingir o número máximo de gerações ou um limiar de aptidão.

    _evaluate_population()
        Avalia a aptidão de todos os cromossomos da população usando a fitness_function.

    _mutate(chromosome, mutation_rate=0.05, learning_rate=0.1)
        Aplica mutação gaussiana a cada gene do cromossomo com uma taxa de mutação.

    _elitism()
        Retorna o melhor cromossomo da população e sua aptidão.

    _select_parent()
        Realiza seleção por torneio e retorna o índice de um dos pais.

    _crossover(parent1, parent2)
        Realiza o _crossover aritmético (média ponderada) entre dois cromossomos.

    _achieved_threshold(threshold=10, mode='max')
        Verifica se a aptidão máxima ou média da população atingiu o limiar definido.
    """

    def __init__(self, pop_size:int, chromosome_size:int, fitness_function, max_iter:int=100, learning_rate:float=0.1, mutation_rate:float=0.1, verbose:bool=False):
        """
        Inicializa o algoritmo genético.
        """
        self._pop_size = pop_size
        self._chromosome_size = chromosome_size
        self._fitness_function = fitness_function
        self._max_iter = max_iter
        self._verbose = verbose
        self._learning_rate = learning_rate
        self._mutation_rate = mutation_rate
        self._n_elite = self._pop_size // 3

        # Inicializa a população com valores aleatórios entre -1 e 1
        self._population = [np.random.uniform(-1, 1, self._chromosome_size).tolist() for _ in range(self._pop_size)]
        self._fitness_scores = [0.0 for _ in range(self._pop_size)]

    def run(self, threshold:float=5000) -> list[float]:
        """
        Executa o ciclo do algoritmo genético até atingir o número máximo de gerações ou o limiar de aptidão.
        """
        for gen in range(0, self._max_iter):
            print(f"\n{'='*10} Geração {gen} {'='*10}")

            self._evaluate_population()

            elites, elite_scores = self._elitism_list()
            print(f'{elite_scores=}')
            new_population = elites.copy()

            for _ in elites:
                p1 = new_population[random.randint(0, len(new_population) - 1)]
                p2 = new_population[random.randint(0, len(new_population) - 1)]
                child = self._crossover(p1, p2)
                self._mutate(child)
                new_population.append(child)

            while len(new_population) < self._pop_size:
                p1 = self._population[self._select_parent()]
                p2 = self._population[self._select_parent()]
                child = self._crossover(p1, p2)
                self._mutate(child)
                new_population.append(child)

            self._population = new_population

            if self._achieved_threshold(threshold=threshold):
                if self._verbose:
                    print(f"GeneticAlgorithm : Atingiu a aptidão desejada : Geração={gen} : Fitness={elite_scores[0]:.2f}")
                break

        print(f"GeneticAlgorithm : Treinamento concluído! Fitness={elite_scores[0]:.2f}")
        return elites[0]

    def _evaluate_population(self, optimized:bool=True) -> None:
        """
        Avalia a aptidão de cada cromossomo da população usando a fitness_function.
        """
        if optimized:
            with Pool() as pool:
                self._fitness_scores = pool.map(self._fitness_function, self._population)
            print(f'{self._fitness_scores=}')
        else:
            for i, chromosome in enumerate(self._population):
                self._fitness_scores[i] = self._fitness_function(chromosome)

        if self._verbose:
            print(f"GeneticAlgorithm : Fitnesses={self._fitness_scores}")

    def _elitism_list(self) -> tuple[list[list[float]], list[float]]:
        # ordena índices por fitness decrescente
        ranked = sorted(range(self._pop_size), key=lambda i: self._fitness_scores[i], reverse=True)
        elites = [ self._population[i] for i in ranked[:self._n_elite] ]
        elite_scores = [ self._fitness_scores[i] for i in ranked[:self._n_elite] ]
        print(f'{elite_scores=}')
        if self._verbose:
            print(f"GeneticAlgorithm : Elites Fitnesses={elite_scores}")
        return elites, elite_scores

    def _elitism(self) -> tuple[list[float], float]:
        """
        Retorna o melhor cromossomo da população e sua aptidão.
        """
        best_idx = np.argmax(self._fitness_scores)
        best_chromosome = self._population[best_idx]
        best_fitness = self._fitness_scores[best_idx]


        if self._verbose:
            print(f"GeneticAlgorithm : Best Chromosome={best_chromosome}")
            print(f"GeneticAlgorithm : Best Fitness={best_fitness:.2f}")
        return best_chromosome, best_fitness

    def _select_parent(self) -> int:
        """
        Seleciona um cromossomo por torneio: escolhe dois aleatórios e retorna o melhor.
        """
        parent1, parent2 = random.sample(range(len(self._population)), 2)

        return parent1 if self._fitness_scores[parent1] > self._fitness_scores[parent2] else parent2

    def _crossover(self, parent1, parent2) -> list[float]:
        """
        Realiza crossover aritmético entre dois cromossomos, gerando um novo filho.
        """
        a = random.uniform(0, 1)
        return [a * p1 + (1 - a) * p2 for p1, p2 in zip(parent1, parent2)]

    def _mutate(self, chromosome:list[float]) -> None:
        """
        Aplica mutação gaussiana em cada gene do cromossomo com uma certa taxa de mutação.
        """
        for i in range(len(chromosome)):
            if random.random() < self._mutation_rate:
                if random.random() < 0.5:
                    chromosome[i] += np.random.normal(0, self._learning_rate)
                else:
                    chromosome[i] -= np.random.normal(0, self._learning_rate)
                chromosome[i] = max(min(chromosome[i], 1), -1)  # Mantém os valores no intervalo [-1, 1]

    def _achieved_threshold(self, threshold:int) -> bool:
        """
        Verifica se a aptidão da população atingiu o limiar definido.
        Parâmetros:
            threshold: valor mínimo de aptidão para considerar como atingido.
            mode: 'max' para considerar o melhor cromossomo, 'avg' para considerar a média da população.
        """
        return max(self._fitness_scores) >= threshold
