# Deps
from functools import partial
import json

# Models
from mlp.genetic_algorithm import GeneticAlgorithm
from mlp.multilayer_perceptron import MultilayerPerceptron
from minimax.minimax import Minimax

# Game
from tictactoe.board import Board as tictactoe

def evaluate_fitness(model: MultilayerPerceptron, minimax: Minimax, chromosome: list) -> float:
    """
    Função de aptidão para o Algoritmo Genético.

    Parâmetros:
    -----------
    model : MultilayerPerceptron
        Rede MLP que será ajustada.

    minimax : Minimax
        Adversário controlado por algoritmo de decisão.

    chromosome : list
        Vetor linear de pesos da rede MLP.

    Retorna:
    --------
    float : valor da aptidão com base no resultado do jogo.
        +10 vitória do MLP
        +1 empate
        -5 derrota
        -20 jogada inválida
    """
    model.load_weights_from_vector(chromosome)
    result = play(model, minimax)

    if 'error' in result and result['error'] == 'p1':
        return -20  # Jogada inválida do MLP

    match result['result']:
        case 3: return 10  # X_WIN (MLP)
        case 1: return 1   # DRAW
        case 0: return -5  # O_WIN (Minimax)
        case _: return 0   # fallback

def play(p1, p2) -> dict:
    """
    Executa uma partida entre dois agentes com método predict().

    Retorna:
    --------
    dict com resultado e estado final do tabuleiro.
    """
    ttt = tictactoe()

    while True:
        if not ttt.update_board(1,p1.predict(ttt.board)):
            return {'error': 'p1', 'reason': 'P1 invalid play', 'result': -1}
        if not ttt.is_ongoing():
            break

        if not ttt.update_board(-1, p2.predict(ttt.board)):
            return {'error': 'p2', 'reason': 'P2 invalid play', 'result': -1}
        if not ttt.is_ongoing():
            break

    return {'board': ttt.board, 'result': ttt.check_win()}


if __name__ == '__main__':
    # Definição da arquitetura da MLP
    TOPOLOGY = [9, 9, 9]
    POPULATION_SIZE = 100
    model = MultilayerPerceptron(TOPOLOGY)
    chromosome_size = model.count_weights()

    # Pipeline de dificuldade: ajuste conforme necessário
    pipeline = [
        2 * 'easy',   # 80% aleatório
        3 * 'medium', # 50% aleatório
        5 * 'hard'    # 0% aleatório (Minimax puro)
    ]

    best_fitness = float('-inf')
    best_chromosome = None

    for mode in pipeline:
        print(f'\n--- Treinando contra adversário no modo: {mode.upper()} ---')
        minimax = Minimax(mode=mode)
        fitness_function = partial(evaluate_fitness, model, minimax)
        ag = GeneticAlgorithm(
            pop_size=POPULATION_SIZE,
            chromosome_size=chromosome_size,
            fitness_function=fitness_function
        )

        print(f'\n--- Executando Algoritmo Genético ---')

        # Executa o AG e retorna o melhor cromossomo
        chromosome, fitness = ag.run()#verbose=True)

        # Atualiza o melhor modelo global
        print(f'\n--- Atualização do modelo ({fitness > best_fitness})---')
        if fitness > best_fitness:
            best_fitness = fitness
            best_chromosome = chromosome
            model.load_weights_from_vector(best_chromosome)

    print(f"\nMelhor fitness alcançado: {best_fitness:.2f}")
    with open('model.json', '+w') as f:
        f.write(json.dumps(model.to_json()))
    print(model.to_json())
