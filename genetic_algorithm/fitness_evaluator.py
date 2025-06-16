from tictactoe.board import Board as tictactoe
from model import IModel


# RODA EM PARALELO! NÃO ADICIONAR PRINTS NESSA CLASSE!
class FitException(Exception):
    pass
class FitnessEvaluator:
    def __init__(self, learner: IModel, trainer: IModel, pipeline:dict[str]):
        self.learner = learner
        self.trainer = trainer
        self.pipeline = pipeline

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
        self.learner.update(chromosome)
        learner_fitness = 0

        for mode in self.pipeline:
            self.trainer.update(mode)
            insights = self._play(self.learner, self.trainer)

            learner_fitness += self._compute_score(insights)
        return learner_fitness

    @staticmethod
    def _compute_score(insights:int) -> float:
        match insights:
            case 1:  return 100  # X_WIN (learner)
            case 0:  return 70   # DRAW
            case -1: return -50  # O_WIN (trainer)
            case -2: return -100 # learner failed!

    @staticmethod
    def _play(learner:IModel, trainer:IModel) -> int | None:
        """
        Executa uma partida entre dois agentes com método predict().

        Retorna:
        --------
        dict com resultado e estado final do tabuleiro.
        """
        ttt = tictactoe()

        while True:
            if not ttt.update_board(1, learner.predict(ttt.board)):
                return -2
            if not ttt.is_ongoing():
                break

            if not ttt.update_board(-1, trainer.predict(ttt.board)):
                raise FitException(f"Trainer has failed to play. {ttt.board}")

            if not ttt.is_ongoing():
                break

        return ttt.check_win()
