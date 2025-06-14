from tictactoe.board import Board as tictactoe
from model import IModel

class FitnessEvaluator:
    def __init__(self, learner: IModel, trainer: IModel, pipeline:dict[str]):
        self.learner = learner
        self.trainer = trainer
        self.pipeline = pipeline

    def __call__(self, chromosome:list[float]):
        return self.evaluate_fitness(chromosome)

    def evaluate_fitness(self, chromosome: list[float]) -> float:
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
            print(f'\n--- Treinando contra adversário no modo: {mode.upper()} ---')

            self.trainer.update(mode)
            _, insights = self._play(self.model, self.minimax)

            learner_fitness += self._compute_score(insights)
        return learner_fitness

    def _compute_score(self, insights:dict) -> float:
        if 'error' in insights and insights['error'] == 'p1':
            return -100  # Jogada inválida do MLP
        match insights['result']:
            case 1: return 100  # X_WIN (MLP)
            case 0: return 70   # DRAW
            case -1: return -50 # O_WIN (Minimax)
            case _: return -100 # fallback

    @staticmethod
    def _play(player1:IModel, player2:IModel) -> dict:
        """
        Executa uma partida entre dois agentes com método predict().

        Retorna:
        --------
        dict com resultado e estado final do tabuleiro.
        """
        ttt = tictactoe()

        while True:
            if not ttt.update_board(1,player1.predict(ttt.board)):
                return ttt.board, {'error': 'p1', 'reason': 'P1 invalid play', 'result': -1}
            if not ttt.is_ongoing():
                break

            if not ttt.update_board(-1, player2.predict(ttt.board)):
                return ttt.board, {'error': 'p2', 'reason': 'P2 invalid play', 'result': -1}
            if not ttt.is_ongoing():
                break

        return ttt.board, {'result': ttt.check_win()}
