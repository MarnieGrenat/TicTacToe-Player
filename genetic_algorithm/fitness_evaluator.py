from tictactoe.board import Board as tictactoe
from model import IModel


# RODA EM PARALELO! NÃO ADICIONAR PRINTS NESSA CLASSE!
class FitException(Exception):
    pass
class FitnessEvaluator:
    def __init__(self, learner: IModel, trainer: IModel, pipeline:dict[str], verbose:bool=False):
        self._learner = learner
        self._trainer = trainer
        self._pipeline = pipeline
        self._verbose = verbose

    def __call__(self, chromosome:list[float]):
        return self._evaluate_fitness(chromosome)

    def _evaluate_fitness(self, chromosome: list[float]) -> float:
        """
        Função de aptidão para o Algoritmo Genético.

        Parâmetros:
        -----------
        chromosome : list[float]
            Vetor linear de pesos da rede MLP.

        Retorna:
        --------
        float : valor da aptidão com base no resultado do jogo.
            +10 vitória do MLP
            +1 empate
            -5 derrota
            -20 jogada inválida
        """
        self._learner.update(chromosome)
        learner_fitness = 0

        for mode in self._pipeline:
            self._trainer.update(mode)
            insights = self._play(self._learner, self._trainer)

            learner_fitness += self._compute_score(insights)

        print(f'FitnessEvaluator : Round de jogadas finalizado. Fitness={learner_fitness}')
        return learner_fitness

    def _play(self, player1:IModel, player2:IModel) -> int | None:
        """
        Executa uma partida entre dois agentes com método predict().

        Retorna:
        --------
        dict com resultado e estado final do tabuleiro.
        """
        ttt = tictactoe()
        if self._verbose:
            print('Starting new round')
        while True:
            p1_play = player1.predict(ttt.board)

            if not ttt.update_board(1, p1_play):
                if self._verbose:
                    print(f'Player 1 : Failed : Prediction={p1_play} : Board={ttt.board}')
                return -2
            else:
                if self._verbose:
                    print(f'Player 1 : Success : Prediction={p1_play} : Board={ttt.board}')

            if not ttt.is_ongoing():
                break

            p2_play = player2.predict(ttt.board)
            if not ttt.update_board(-1, p2_play):
                if self._verbose:
                    print(f'Player 2 : Fail : Prediction={p1_play} : Board={ttt.board}')
                raise FitException(f"Player 2 has failed to play. Prediction={p2_play} : Board={ttt.board}")
            else:
                if self._verbose:
                    print(f'Player 2 : Success : Prediction={p1_play} : Board={ttt.board}')

            if not ttt.is_ongoing():
                break

        return ttt.check_win()

    def _compute_score(self, insights:int) -> float:
        match insights:
            case 1:  return 100  # X_WIN (learner)
            case 0:  return 70   # DRAW
            case -1: return -50  # O_WIN (trainer)
            case -2: return -100 # learner failed!
        raise FitException(f"Unexpected Output={insights}")
