class Board:
    O_WIN   = O = 0
    DRAW    = 1
    ONGOING = 2
    X_WIN   = X = 3

    def __init__(self):
        self.board = self.clear()

    def clear(self):
        return [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

    def reset(self):
        self.board = self.clear()

    def update_board(self, symbol: int, x: int, y: int) -> bool:
        '''update board status. | symbol= int (-1 or 1) | x= coordinate x between 0 and 2 | y= coordinate y between 0 and 2'''
        if (self.board[x][y] != '0') and self.__valid_coordinates(x, y) and self.__valid_symbol(symbol):
            self.board[x][y] = symbol
            return True
        return False

    def export_board(self) -> list[list[str]]:
        '''Export board containing visual symbols (prettify)'''
        result = [['', '', ''],
                  ['', '', ''],
                  ['', '', '']]

        for x in range(len(result)):
            for y in range(len(result)):
                match self.board[x][y]:
                    case -1 : result[x][y] = 'O'
                    case  1 : result[x][y] = 'X'
        return result

    def export_board_raw(self):
        return self.board

    def flatten_board(self):
        ''' Transforma as observações 3x3 em 1x9'''
        flat = [cell for row in self.board for cell in row]
        return flat

    def check_wins(self):
        '''Verifica o estado do tabuleiro e o retorna'''
        # Diagonais
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self._get_label(self.board[1][1])
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self._get_label(self.board[1][1])

        # Linhas e colunas
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                return self._get_label(self.board[0][i])
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return self._get_label(self.board[i][0])

        # Verificar se o jogo ainda está em andamento (se há posições vazias)
        if 0 in self.flatten_board():
            return 2  # Jogo em andamento

        # Verificar se o jogo terminou em empate (sem espaços vazios e sem vencedor)
        return 1  # Empate

    def _get_label(self, symbol : int) -> int:
        match symbol:
            case -1: return 0
            case 1: return 3

    def __valid_coordinates(self, x : int, y : int) -> bool:
        return (x < 3) and (x >= 0) and (y < 3) and (y >= 0)

    def __valid_symbol(self, symbol : str) -> bool:
        return symbol in [-1, 1]
