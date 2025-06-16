import random
import numpy as np

# Optimization
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

    def __init__(self, pop_size:int, chromosome_size:int, fitness_function, max_iter:int=100, learning_rate:float=0.1, verbose:bool=False):
        """
        Inicializa o algoritmo genético.
        """
        self._pop_size = pop_size
        self._chromosome_size = chromosome_size
        self._fitness_function = fitness_function
        self._max_iter = max_iter
        self._verbose = verbose
        self._learning_rate = learning_rate

        # Inicializa a população com valores aleatórios entre -1 e 1
        self._population = [np.random.uniform(-1, 1, self._chromosome_size).tolist() for _ in range(self._pop_size)]
        self._fitness_scores = [0.0 for _ in range(self._pop_size)]

    def run(self, threshold:float=5000) -> None:
        """
        Executa o ciclo do algoritmo genético até atingir o número máximo de gerações ou o limiar de aptidão.
        """
        for gen in range(1, self._max_iter + 1):
            print(f"\n{'='*10} Geração {gen} {'='*10}")

            self._evaluate_population()

            chromosome, fitness = self._elitism()
            new_population = [chromosome]

            while len(new_population) < self._pop_size:
                parent1 = self._population[self._select_parent()]
                parent2 = self._population[self._select_parent()]
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                new_population.append(child)
            self._population = new_population

            if self._achieved_threshold(threshold=threshold):
                if self._verbose:
                    print(f"\nAtingiu a aptidão desejada na geração {gen}")
                break

        print(f"\nTreinamento concluído! Melhor aptidão final: {fitness:.2f}")

    def _evaluate_population(self, optimized:bool=True) -> None:
        """
        Avalia a aptidão de cada cromossomo da população usando a fitness_function.
        """
        if optimized:
            with Pool() as pool:
                self._fitness_scores = pool.map(self._fitness_function, self._population)
        else:
            for i, chromosome in enumerate(self._population):
                self._fitness_scores[i] = self._fitness_function(chromosome)

        if self._verbose:
            print(f"Aptidões: {self._fitness_scores}")

    def _elitism(self) -> tuple[list[float], float]:
        """
        Retorna o melhor cromossomo da população e sua aptidão.
        """
        best_idx = np.argmax(self._fitness_scores)
        best_chromosome = self._population[best_idx]
        best_fitness = self._fitness_scores[best_idx]


        if self._verbose:
            print(f"Melhor cromossomo: {best_chromosome}")
            print(f"Melhor aptidão: {best_fitness:.2f}")
        return best_chromosome, best_fitness

    def _select_parent(self) -> int:
        """
        Seleciona um cromossomo por torneio: escolhe dois aleatórios e retorna o melhor.
        """
        parent1, parent2 = random.sample(range(self._pop_size), 2)
        return parent1 if self._fitness_scores[parent1] > self._fitness_scores[parent2] else parent2

    def _crossover(self, parent1, parent2) -> list[float]:
        """
        Realiza crossover aritmético entre dois cromossomos, gerando um novo filho.
        """
        a = random.uniform(0, 1)
        return [a * p1 + (1 - a) * p2 for p1, p2 in zip(parent1, parent2)]

    def _mutate(self, chromosome:list[float], mutation_rate:float=0.05) -> None:
        """
        Aplica mutação gaussiana em cada gene do cromossomo com uma certa taxa de mutação.
        """
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] += np.random.normal(0, self._learning_rate)
                chromosome[i] = max(min(chromosome[i], 1), -1)  # Mantém os valores no intervalo [-1, 1]

    def _achieved_threshold(self, threshold:int) -> bool:
        """
        Verifica se a aptidão da população atingiu o limiar definido.
        Parâmetros:
            threshold: valor mínimo de aptidão para considerar como atingido.
            mode: 'max' para considerar o melhor cromossomo, 'avg' para considerar a média da população.
        """
        return max(self._fitness_scores) >= threshold
