class Board:
    """
    Classe para gerenciar o estado de um tabuleiro de jogo da velha (Tic-Tac-Toe).

    A lógica é baseada em:
    - X representado pelo valor 1 (MLP).
    - O representado pelo valor -1 (Minimax).
    - Posições vazias representadas por 0.

    Estados do jogo:
    ----------------
    X_WIN (3) : X venceu.
    O_WIN (0) : O venceu.
    DRAW (1)  : Empate.
    ONGOING (2) : Jogo em andamento.

    Métodos:
    --------
    update_board(symbol: int, x: int, y: int) -> bool
        Atualiza o tabuleiro com a jogada do jogador no ponto (x, y). Retorna True se a jogada for válida.

    flatten_board() -> list
        Retorna o tabuleiro como uma lista linear (1x9) para entrada da MLP.

    check_wins() -> int
        Verifica o estado atual do jogo: vitória (X ou O), empate ou jogo em andamento.
    """

    def __init__(self):
        """
        Inicializa o tabuleiro vazio (1x9).
        """
        self.board = [0, 0, 0,
                      0, 0, 0,
                      0, 0, 0]

    def update_board(self, symbol: int, index: int) -> bool:
        """
        Atualiza o tabuleiro com a jogada do jogador.

        Parâmetros:
        -----------
        symbol : int
            -1 para O, 1 para X.
        index : int
            Coordenada da linha (0 a 8).

        Retorna:
        --------
        bool : True se a jogada foi válida, False caso contrário.
        """
        if self.board[index] == 0 and self.__valid_coordinates(index) and self.__valid_symbol(symbol):
            self.board[index] = symbol
            return True
        return False

    def check_win(self) -> int:
        """
        Verifica o estado atual do tabuleiro (representado como lista linear 1x9).

        Retorna:
        --------
        int : Código do estado do jogo.
            3 : X venceu
            0 : O venceu
            1 : Empate
            2 : Jogo em andamento
        """
        b = self.board

        for i in [0, 3, 6]:
            if b[i] == b[i + 1] == b[i + 2] != 0:
                return self._get_label(b[i])
        for i in [0, 1, 2]:
            if b[i] == b[i + 3] == b[i + 6] != 0:
                return self._get_label(b[i])

        # Verifica diagonais
        if b[0] == b[4] == b[8] != 0:
            return self._get_label(b[4])
        if b[2] == b[4] == b[6] != 0:
            return self._get_label(b[4])

        if 0 in b:
            return 2 # Em progesso
        return 1 # Empate

    def is_ongoing(self) -> bool:
        return self.check_win() == 0

    def _get_label(self, symbol: int) -> int:
        """
        Converte o símbolo (-1 para O, 1 para X) em código de vitória.
        """
        return Board.O_WIN if symbol == -1 else Board.X_WIN

    def __valid_coordinates(self, index: int) -> bool:
        """
        Verifica se as coordenadas (x, y) estão dentro do tabuleiro.
        """
        return 0 <= index < 8

    def __valid_symbol(self, symbol: int) -> bool:
        """
        Verifica se o símbolo é válido (-1 para O, 1 para X).
        """
        return symbol in [-1, 1]
