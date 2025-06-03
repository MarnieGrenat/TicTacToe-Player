class Minimax():
    def __init__(self, mode:str='medium'):
        if mode == 'hard' : randomness = 0.0
        elif mode == 'medium' : randomness = 0.5
        elif mode == 'easy' : randomness = 0.8
        else : raise ValueError(f"Unexpected mode : {mode=}")

        # TODO: Implementar Minimax