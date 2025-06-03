# Deps
from functools import partial

# Models
from mlp.multilayer_perceptron import GeneticAlgorithm
from mlp.multilayer_perceptron import MultilayerPerceptron
from minimax.minimax import Minimax

# Case
from tictactoe import board as tictactoe

def evaluate_fitness(model: MultilayerPerceptron, minimax: Minimax, chromosome: list) -> float:
    """
    Função de aptidão para o Algoritmo Genético.

    Recebe:
    -------
    model : MultilayerPerceptron
        Instância da rede MLP a ser otimizada.
    minimax : Minimax
        Instância do adversário (Minimax) para jogar contra.
    chromosome : list
        Vetor linear de pesos (cromossomo) para carregar na MLP.

    Retorna:
    --------
    float : Valor da aptidão calculada com base no resultado da partida.
        +10 para vitória da MLP (X),
        +1 para empate,
        -5 para derrota,
        -20 para jogada inválida.
    """
    model.load_weights_from_vector(chromosome)
    result = play(model, minimax)

    if 'error' in result and result['error'] == 'p1':
        return -20  # Penalização para jogadas inválidas

    match result['result']:
        case 3: return 10  # X_WIN
        case 1: return 1   # DRAW
        case 0: return -5  # O_WIN
        case _: return 0   # Resultado inesperado (fallback)

def play(p1, p2) -> dict:
    """
    Simula uma partida de jogo da velha entre dois jogadores: p1 (MLP) e p2 (Minimax).

    Recebe:
    -------
    p1, p2 : objetos com método predict()
        Agentes para jogar a partida. Devem implementar o método predict(board).

    Retorna:
    --------
    dict : Resultado da partida, incluindo o estado final do tabuleiro e o resultado.
        {'board': estado final, 'result': código do resultado}
        Códigos: 3 (X_WIN), 1 (DRAW), 0 (O_WIN), ou erro.
    """
    ttt = tictactoe()  # Inicializa o tabuleiro

    while True:
        if not ttt.update_board(p1.predict(ttt.flatten_board()), ttt.X):
            return {'error': 'p1', 'reason': 'P1 invalid play', 'result': -1}
        if ttt.check_win() != ttt.ONGOING:
            break

        if not ttt.update_board(p2.predict(ttt.flatten_board()), ttt.O):
            return {'error': 'p2', 'reason': 'P2 invalid play', 'result': -1}
        if ttt.check_win() != ttt.ONGOING:
            break

    return {'board': ttt.flatten_board(), 'result': ttt.check_win()}

if __name__ == '__main__':
    # Definição da topologia da MLP: [9 inputs, 9 hidden, 9 output]
    TOPOLOGY = [9, 9, 9]
    POPULATION_SIZE = 100

    model = MultilayerPerceptron(TOPOLOGY)
    chromosome_size = model.count_weights()
    pipeline = [
        3 * 'easy',
        4 * 'medium',
        1 * 'hard'
    ]

    for mode in pipeline:
        minimax = Minimax(mode)
        # Define a função de aptidão para o AG usando partial para fixar os argumentos model e minimax
        fitness_function = partial(evaluate_fitness, model, minimax)

        ag = GeneticAlgorithm(
            pop_size=POPULATION_SIZE,
            chromosome_size=chromosome_size,
            fitness_function=fitness_function
        )
        # Executa o ciclo do AG até atingir o limiar ou o número máximo de gerações
        ag.run()

    # Persistir o modelo
    model.to_json()
