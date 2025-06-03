from random import randint
import numpy as np

# Funções de dependência
class Genetic_algorithm:
    def __init__(self, rows:int, columns:int, vector:list, max_iter=100):
        self.r = rows
        self.c = columns
        self.v = vector
        self.max = max_iter

        self.pop = self.random_init

    def fit(self):
        counter = 0
        while True:
            # Inicialização
            counter += 1
            pop_int = np.zeros(shape=[self.r, self.c], dtype=int)
            print(f'{5 * '='}Geracao {counter}{5 * '='}')

            ## Calculando o algoritmo
            self.calcula_aptidao(True)

            i = self.elitismo(True)
            pop_int[0] = self.pop[i].copy()

            self.crossover(pop_int)
            self.pop = pop_int

            self.mutacao()

            # Condições de parada
            if counter >= self.max or self.aptidao_eh_boa(.95):
                break

        print(f'{5 * '='}Resultado da Geração {counter}{5 * '='}')
        aptidoes = [int(e[-1]) for e in self.pop]
        print(f'Aptidão : Aptidões={aptidoes}')

    def calcula_aptidao(self, verbose:bool=False):
        for i in range(len(self.pop)):
            acumulador_0 = 0
            acumulador_1 = 0
            for j in (range(len(self.pop[0]) - 1)):
                # Calcula aptidões
                if self.pop[i][j] == 0:
                    acumulador_0 += self.v[j]
                else:
                    acumulador_1 += self.v[j]
            # Preenche ultima coluna com aptidões
            self.pop[i][-1] = abs(acumulador_0 - acumulador_1)
        if verbose:
            print(f'Aptidão : Todas Aptidões={[int(e[-1]) for e in self.pop]}')

    def mutacao(self, porc=50):
        '''Gera mutacao n% das vezes. n=porc'''
        if randint(0, 100) < porc:
            return # evita executar operação porc% das vezes
        # acesso linha random
        i = randint(1, len(self.pop) - 1)
        j = randint(0, len(self.pop[0]) - 1)

        # shifta valor (se era 0, vira 1. se era 1, vira 0)
        self.pop[i][j] = abs(self.pop[i][j] - 1)

    def elitismo(self, verbose:bool=False):
        '''Retorna índice do menor elitismo presente na população'''
        _ = 0
        m = 9999999
        for _ in range(len(self.pop)):
            if self.pop[_][-1] < m:
                m = self.pop[_][-1]
                i = _
        if verbose:
            print(f'Elitismo : Indice={i} : Aptidão={self.pop[i][-1]}')
        return i

    def torneio(self):
        '''Retorna o índice da observação de melhor aptidão'''
        idx1 = randint(0, len(self.pop)-1)
        idx2 = randint(0, len(self.pop)-1)
        # Evita q idx1 == idx2
        while idx1 == idx2:
            idx2 = randint(0, len(self.pop)-1)

        # Retorna índice do menor valor
        if self.pop[idx1][-1] < self.pop[idx2][-1]:
            return idx1
        return idx2

    def crossover(self, interm):
        '''Cruza indivíduos, atualizando filhos'''
        primeira_metade = len(self.pop[0])//2
        ultima_metade =  len(self.pop[0] - 1) # Ignora a coluna de aptidões

        for i in (range(1, len(self.pop), 2)): # Pula de 2 em 2
            mae = self.torneio
            pai = self.torneio

            # FIXME: Não funciona pro problema atual. Utilizar média?
            for j in range(primeira_metade): # até primeira metade
                interm[i][j] = self.pop[mae][j]
                interm[i+1][j] = self.pop[pai][j]

            for j in range(primeira_metade, ultima_metade): # da primeira metade até ultima metade
                interm[i][j] = self.pop[pai][j]
                interm[i+1][j] = self.pop[mae][j]

    def aptidao_eh_muito_boa(self, porcentagem:float=.90):
        '''Retorna True que a porcentagem de aptidões igual a zero for maior que 90% (ou valor do param porcentagem)'''
        aptidao_0 = sum(1 for linha in self.pop if linha[-1] == 0)
        aptidao_todas = len(self.pop)
        return (aptidao_0/aptidao_todas) >= porcentagem

    def random_init(self):
        '''Inicializa uma matriz de row linhas e col colunas preenchida aleatoriamente com 0's e 1's '''
        m = np.zeros(shape=[self.r, self.c], dtype=int)
        for i in range(self.r):
            for j in range(self.c):
                m[i][j]= randint(0,1)
        return m
