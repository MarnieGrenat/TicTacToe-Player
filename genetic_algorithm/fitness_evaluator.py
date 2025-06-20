from .tictactoe.board import Board as tictactoe
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
            insights, board = self._play(self._learner, self._trainer)

            learner_fitness += self._compute_score(mode, insights)
        print(f'FitnessEvaluator : Round de jogadas finalizado. Fitness={learner_fitness} : Board={board}')

        return learner_fitness

    def _play(self, player1:IModel, player2:IModel) -> tuple[int, list[int]] | None:
        """
        Executa uma partida entre dois agentes com método predict().

        Retorna:
        --------
        dict com resultado e estado final do tabuleiro.
        """
        ttt = tictactoe()
        if self._verbose:
            print('FitnessEvaluator : Starting new round')
        while True:
            p1_play = player1.predict(ttt.board)
            if not ttt.update_board(1, p1_play):
                if self._verbose:
                    print(f'FitnessEvaluator : Player 1 : Failed : Prediction={p1_play} : Board={ttt.board}')
                return -2, ttt.board
            if not ttt.is_ongoing():
                break

            p2_play = player2.predict(ttt.board)
            if not ttt.update_board(-1, p2_play):
                raise FitException(f"FitnessEvaluator : Player 2 has failed to play. Prediction={p2_play} : Board={ttt.board}")
            if not ttt.is_ongoing():
                break

        return ttt.check_win(), ttt.board

    def _compute_score(self, mode: str, insights: int) -> float:
        if mode == 'easy':
            good = 1
            bad = 3
        elif mode == 'medium':
            good = 2
            bad = 2
        else:
            good = 3
            bad = 1
        match insights:
            case 1 : return good * +200     # X venceu
            case 0 : return good * +60     # Empate
            case -1: return bad * -500   # O venceu
            case -2: return -50000  # Jogada inválida
        raise FitException(f"FitnessEvaluator : Unexpected Output={insights}")

    @staticmethod
    def test_model(learner: IModel, trainer: IModel, rounds: int = 50):
        results = {"win": 0, "draw": 0, "loss": 0, "mlp_fail": 0, "minimax_fail": 0}

        for _ in range(rounds):
            board = tictactoe()

            while board.is_ongoing():
                p1_move = learner.predict(board.board)
                if not board.update_board(1, p1_move):
                    results["mlp_fail"] += 1
                    print(f'MLP FAIL : {board.board}')
                    break

                if not board.is_ongoing():
                    if board.check_win() == 1:
                        results["win"] += 1
                        print(f'Win : {board.board}')
                    break

                p2_move = trainer.predict(board.board)
                if not board.update_board(-1, p2_move):
                    results["minimax_fail"] += 1
                    print(f'Minimax FAIL : {board.board}')
                    break

            else:  # ✅ Executa se não houve break (jogo terminou normalmente)
                result = board.check_win()
                if result == 1:
                    results["win"] += 1
                    print(f'Win : {board.board}')
                elif result == 0:
                    results["draw"] += 1
                    print(f'Draw : {board.board}')
                elif result == -1:
                    results["loss"] += 1
                    print(f'Loss : {board.board}')

        print(f"FitnessEvaluator : Results over {rounds} games: {results}")


