import random
from .model_interface import IModel


class Minimax(IModel):
    """
    Jogador automático usando Minimax com dificuldade ajustável e otimização via poda alfa-beta.
    """

    def __init__(self):
        self.mode = 'medium'
        self.update('medium')

    def update(self, mode: str = 'medium') -> None:
        """
        Atualiza a dificuldade do Minimax.
        """
        if mode == 'hard':
            self.randomness = 0.0
        elif mode == 'medium':
            self.randomness = 0.5
        elif mode == 'easy':
            self.randomness = 0.8
        else:
            raise ValueError(f"Minimax : Modo inválido: {mode}. Use 'easy', 'medium' ou 'hard'.")
        self.mode = mode

    def predict(self, board: list) -> int:
        """
        Escolhe uma jogada no tabuleiro (índice de 0 a 8).
        """
        empty_indices = [i for i, val in enumerate(board) if val == 0]

        if not empty_indices:
            raise ValueError(f"Minimax : [ERROR] No possible moves: {board}")

        # Se cair na aleatoriedade da dificuldade, joga aleatório
        if random.random() < self.randomness:
            return random.choice(empty_indices)

        # Executa Minimax com poda alfa-beta
        best_score = float('inf')
        best_move = None

        for idx in empty_indices:
            new_board = board.copy()
            new_board[idx] = -1  # Minimax joga como O
            score = self.minimax(new_board, maximizing=True, alpha=float('-inf'), beta=float('inf'))
            if score < best_score:
                best_score = score
                best_move = idx

        if best_move is None:
            # Fallback defensivo (não deveria acontecer)
            return random.choice(empty_indices)

        return best_move

    def minimax(self, board, maximizing: bool, alpha: float, beta: float) -> int:
        """
        Algoritmo Minimax com poda alfa-beta.
        """
        winner = self.check_winner(board)
        if winner is not None:
            return winner

        if maximizing:
            max_eval = float('-inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = 1  # MLP joga (Maximiza)
                    eval = self.minimax(board, False, alpha, beta)
                    board[i] = 0
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  # Corta
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = -1  # Minimax joga (Minimiza)
                    eval = self.minimax(board, True, alpha, beta)
                    board[i] = 0
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Corta
            return min_eval

    def check_winner(self, board: list):
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for combo in wins:
            values = [board[i] for i in combo]
            if values == [1, 1, 1]:
                return 1
            elif values == [-1, -1, -1]:
                return -1

        if 0 not in board:
            return 0  # Empate

        return 2  # Jogo em andamento

