import random
import numpy as np

class Genetic_algorithm:
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
    run(threshold=9.5, verbose=False)
        Executa o ciclo do algoritmo genético até atingir o número máximo de gerações ou um limiar de aptidão.

    evaluate_population(verbose=False)
        Avalia a aptidão de todos os cromossomos da população usando a fitness_function.

    mutate(chromosome, mutation_rate=0.05, learning_rate=0.1)
        Aplica mutação gaussiana a cada gene do cromossomo com uma taxa de mutação.

    elitism(verbose=False)
        Retorna o melhor cromossomo da população e sua aptidão.

    select_parent()
        Realiza seleção por torneio e retorna o índice de um dos pais.

    crossover(parent1, parent2)
        Realiza o crossover aritmético (média ponderada) entre dois cromossomos.

    achieved_threshold(threshold=10, mode='max')
        Verifica se a aptidão máxima ou média da população atingiu o limiar definido.
    """

    def __init__(self, pop_size: int, chromosome_size: int, fitness_function, max_iter=100):
        """
        Inicializa o algoritmo genético.
        """
        self.pop_size = pop_size
        self.chromosome_size = chromosome_size
        self.fitness_function = fitness_function
        self.max_iter = max_iter

        # Inicializa a população com valores aleatórios entre -1 e 1
        self.population = [np.random.uniform(-1, 1, self.chromosome_size).tolist() for _ in range(self.pop_size)]
        self.fitness_scores = [0.0 for _ in range(self.pop_size)]

    def evaluate_population(self, verbose=False):
        """
        Avalia a aptidão de cada cromossomo da população usando a fitness_function.
        """
        for i, chromosome in enumerate(self.population):
            self.fitness_scores[i] = self.fitness_function(chromosome)
        if verbose:
            print(f"Aptidões: {self.fitness_scores}")

    def mutate(self, chromosome, mutation_rate=0.05, learning_rate=0.1):
        """
        Aplica mutação gaussiana em cada gene do cromossomo com uma certa taxa de mutação.
        """
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] += np.random.normal(0, learning_rate)
                chromosome[i] = max(min(chromosome[i], 1), -1)  # Mantém os valores no intervalo [-1, 1]

    def elitism(self, verbose: bool = False):
        """
        Retorna o melhor cromossomo da população e sua aptidão.
        """
        best_idx = np.argmax(self.fitness_scores)
        best_chromosome = self.population[best_idx].copy()
        best_fitness = self.fitness_scores[best_idx]
        if verbose:
            print(f"Melhor aptidão: {best_fitness:.2f}")
        return best_chromosome, best_fitness

    def select_parent(self) -> int:
        """
        Seleciona um cromossomo por torneio: escolhe dois aleatórios e retorna o melhor.
        """
        parent1, parent2 = random.sample(range(self.pop_size), 2)
        return parent1 if self.fitness_scores[parent1] > self.fitness_scores[parent2] else parent2

    def crossover(self, parent1, parent2):
        """
        Realiza crossover aritmético entre dois cromossomos, gerando um novo filho.
        """
        a = random.uniform(0, 1)
        return [a * p1 + (1 - a) * p2 for p1, p2 in zip(parent1, parent2)]

    def achieved_threshold(self, threshold: float = 10, mode: str = 'max') -> bool:
        """
        Verifica se a aptidão da população atingiu o limiar definido.
        Parâmetros:
            threshold: valor mínimo de aptidão para considerar como atingido.
            mode: 'max' para considerar o melhor cromossomo, 'avg' para considerar a média da população.
        """
        if mode == 'avg':
            return (sum(self.fitness_scores) / len(self.fitness_scores)) >= threshold
        else:
            return (max(self.fitness_scores)) >= threshold

    def run(self, threshold=9.5, verbose=False):
        """
        Executa o ciclo do algoritmo genético até atingir o número máximo de gerações ou o limiar de aptidão.
        """
        for gen in range(1, self.max_iter + 1):
            print(f"\n{'='*10} Geração {gen} {'='*10}")

            self.evaluate_population(verbose)

            chromosome, fitness = self.elitism(verbose)
            new_population = [chromosome]

            while len(new_population) < self.pop_size:
                parent1 = self.population[self.select_parent()]
                parent2 = self.population[self.select_parent()]
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population

            if self.achieved_threshold(threshold=threshold):
                if verbose:
                    print(f"\nAtingiu a aptidão desejada na geração {gen}")
                break

        if verbose:
            print(f"\nTreinamento concluído! Melhor aptidão final: {fitness:.2f}")
