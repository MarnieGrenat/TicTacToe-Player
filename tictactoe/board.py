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

    # Estados possíveis do jogo
    O_WIN   = 0
    DRAW    = 1
    ONGOING = 2
    X_WIN   = 3

    def __init__(self):
        """
        Inicializa o tabuleiro vazio (3x3).
        """
        self.board = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]

    def update_board(self, symbol: int, x: int, y: int) -> bool:
        """
        Atualiza o tabuleiro com a jogada do jogador.

        Parâmetros:
        -----------
        symbol : int
            -1 para O, 1 para X.
        x : int
            Coordenada da linha (0 a 2).
        y : int
            Coordenada da coluna (0 a 2).

        Retorna:
        --------
        bool : True se a jogada foi válida, False caso contrário.
        """
        if self.board[x][y] == 0 and self.__valid_coordinates(x, y) and self.__valid_symbol(symbol):
            self.board[x][y] = symbol
            return True
        return False

    def flatten_board(self) -> list:
        """
        Retorna o tabuleiro como uma lista linear (1x9), para entrada da rede MLP.
        """
        return [cell for row in self.board for cell in row]

    def check_wins(self) -> int:
        """
        Verifica o estado atual do tabuleiro.

        Retorna:
        --------
        int : Código do estado do jogo.
            3 : X venceu
            0 : O venceu
            1 : Empate
            2 : Jogo em andamento
        """
        # Verifica diagonais
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self._get_label(self.board[1][1])
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self._get_label(self.board[1][1])

        # Verifica linhas e colunas
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return self._get_label(self.board[i][0])
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                return self._get_label(self.board[0][i])

        # Verifica se há espaços vazios (jogo em andamento)
        if 0 in self.flatten_board():
            return Board.ONGOING

        # Se não houver vencedor nem espaços vazios, é empate
        return Board.DRAW

    def _get_label(self, symbol: int) -> int:
        """
        Converte o símbolo (-1 para O, 1 para X) em código de vitória.
        """
        return Board.O_WIN if symbol == -1 else Board.X_WIN

    def __valid_coordinates(self, x: int, y: int) -> bool:
        """
        Verifica se as coordenadas (x, y) estão dentro do tabuleiro.
        """
        return 0 <= x < 3 and 0 <= y < 3

    def __valid_symbol(self, symbol: int) -> bool:
        """
        Verifica se o símbolo é válido (-1 para O, 1 para X).
        """
        return symbol in [-1, 1]
