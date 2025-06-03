from mlp import genetic_algorithm, multilayer_perceptron
from tictactoe import board as tictactoe

def main(topology:list):
    ttt = tictactoe()
    model = multilayer_perceptron(topology) # , ttt.flatten_board())
    minimax = multilayer_perceptron(topology) #FIXME

    while ttt.check_win() == ttt.ONGOING:
        # MLP play
        guess = model.predict(ttt.flatten_board())
        ttt.update_board(guess, ttt.X)

        # if MLP didn't win
        if ttt.check_win() != ttt.ONGOING:
            break

        # Minimax play
        minimax.predict(ttt.flatten_board())
        ttt.update_board(guess, ttt.O)

    # Ou alguém venceu, ou ninguém venceu
    print(ttt.check_win())
    print(ttt.flatten_board())


if __name__ == '__main__':
    TOPOLOGY = [9, 9, 9]
    main(TOPOLOGY)
