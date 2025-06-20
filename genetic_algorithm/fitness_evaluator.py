from ..tictactoe.board import Board as tictactoe
from ..model import IModel


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

    def _compute_score(self, insights: int) -> float:
        match insights:
            case -1: return 150   # X venceu
            case 0:  return 30    # Empate
            case 1:  return -50   # O venceu
            case -2: return -100  # Jogada inválida
        raise FitException(f"FitnessEvaluator : Unexpected Output={insights}")

    @staticmethod
    def test_model(learner: IModel, trainer: IModel, rounds: int = 50):
        results = {"win": 0, "draw": 0, "loss": 0}
        print()

        for _ in range(rounds):
            board = tictactoe()

            while board.is_ongoing():
                p1_move = learner.predict(board.board)
                if not board.update_board(1, p1_move):
                    # Jogada inválida — considera derrota para MLP
                    results["loss"] += 1
                    break

                if not board.is_ongoing():
                    break

                p2_move = trainer.predict(board.board)
                if not board.update_board(-1, p2_move):
                    # Jogada inválida do minimax — considera vitória para MLP
                    results["win"] += 1
                    break

            else:
                # Se o loop não foi quebrado por jogadas inválidas:
                result = board.check_win()
                if result == -1:
                    results["win"] += 1
                elif result == 0:
                    results["draw"] += 1
                elif result == 1:
                    results["loss"] += 1

        print(f"FitnessEvaluator : Results over {rounds} games: {results}")

