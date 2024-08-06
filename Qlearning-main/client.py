import connection as cn
import numpy as np
import random
from connection import *

s = cn.connect(2037)

# Hyperparameters
alpha = 0.15  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.15  # Exploration rate
num_episodios = 1000

acoes = ['left', 'right', 'jump']

Q = np.zeros((24 * 4, len(acoes))) # 24 plataformas * 4 direções = 96 estados

def converter_estado(estado):
    """Converte a string binária em um índice na Q table"""
    plataforma = int(estado[2:7], 2) # Plataformas 0-23
    direcao = int(estado[7:], 2)     # Direções  00, 01, 10, 11
    return plataforma * 4 + direcao

# decaimento exponencial do alpha
def definir_alpha(episodio):
    return max(0.150, 0.8*(1/(1+0.008*episodio)))

# decaimento exponencial do eps
def definir_epsilon(episodio):
    return max(0.075, 0.5*(1/(1+0.008*episodio)))

def escolher_acao(estado, episodio):
    epsilon = definir_epsilon(episodio)
    """Estratégia epsilon-greedy"""
    if random.uniform(0, 1) < epsilon:
        return random.choice(acoes)
    else:
        indice_estado = converter_estado(estado)
        return acoes[np.argmax(Q[indice_estado])]

def q_table(estado, acao, reward, prox_estado, Q, episodio):
    """Atualiza a Q table baseado na experiência do agente."""
    indice_estado = converter_estado(estado)
    indice_prox_estado = converter_estado(prox_estado)
    indice_acao = acoes.index(acao)
    melhor_prox_acao = np.max(Q[indice_prox_estado])
    alpha = definir_alpha(episodio)
    Q[indice_estado, indice_acao] += alpha * (reward + gamma * melhor_prox_acao - Q[indice_estado, indice_acao])
    return Q

try:
    for episodio in range(num_episodios):
        estado = '0b0000000' # estado inicial, piso 0, direção norte
        while True:
            acao = escolher_acao(estado, episodio)
            prox_estado, reward = get_state_reward(s, acao)
            Q = q_table(estado, acao, reward, prox_estado, Q, episodio)
            if reward == 300 or reward == -100:
                break  # Episodio acaba se o agente chegar no destino ou cair
            estado = prox_estado
        print(f'Episode: {episodio}; Reward: {reward}')
except KeyboardInterrupt:
    np.savetxt("resultado.txt", Q.astype(np.float16))
np.savetxt("resultado.txt", Q.astype(np.float16))

print("Treinamento completado!")

