import random

class Minimax:
    """
    Jogador automático usando o algoritmo Minimax com dificuldade ajustável.

    Parâmetros:
    -----------
    mode : str
        Define a dificuldade. Pode ser:
        - 'hard'   : sempre joga a melhor jogada (Minimax 100%)
        - 'medium' : 50% de chance de jogar aleatoriamente
        - 'easy'   : 80% de chance de jogar aleatoriamente
    """

    def __init__(self, mode='medium'):
        if mode == 'hard':
            self.randomness = 0.0
        elif mode == 'medium':
            self.randomness = 0.5
        elif mode == 'easy':
            self.randomness = 0.8
        else:
            raise ValueError(f"Modo inválido: {mode}. Use 'easy', 'medium' ou 'hard'.")

        self.mode = mode

    def predict(self, board: list) -> int:
        """
        Retorna o índice (0 a 8) da jogada escolhida.
        Com probabilidade proporcional à dificuldade, joga aleatoriamente.
        """
        empty_indices = [i for i, val in enumerate(board) if val == 0]

        # Aleatoriedade baseada na dificuldade
        if random.random() < self.randomness:
            return random.choice(empty_indices)

        # Estratégia Minimax (modo hard)
        best_score = float('inf')
        best_move = None

        for idx in empty_indices:
            new_board = board.copy()
            new_board[idx] = -1  # O (Minimax)
            score = self.minimax(new_board, maximizing=True)
            if score < best_score:
                best_score = score
                best_move = idx

        return best_move if best_move is not None else random.choice(empty_indices)

    def minimax(self, board, maximizing: bool) -> int:
        """
        Algoritmo recursivo do Minimax.
        """
        winner = self.check_winner(board)
        if winner is not None:
            return winner

        if maximizing:
            max_eval = float('-inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = 1  # Jogador MLP
                    eval = self.minimax(board, False)
                    board[i] = 0
                    max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = -1  # Jogador Minimax
                    eval = self.minimax(board, True)
                    board[i] = 0
                    min_eval = min(min_eval, eval)
            return min_eval

    def check_winner(self, board: list):
        """
        Verifica o vencedor a partir de um tabuleiro linearizado.

        Retorna:
            +10 se X (MLP) venceu
            -10 se O (Minimax) venceu
            0 se empate
            None se jogo ainda está em andamento
        """
        wins = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6]
        ]
        for combo in wins:
            values = [board[i] for i in combo]
            if values == [1, 1, 1]:
                return +10
            elif values == [-1, -1, -1]:
                return -10

        if 0 not in board:
            return 0  # empate

        return None  # jogo em andamento
