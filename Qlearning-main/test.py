import connection as cn
import numpy as np
import random
from connection import *

Q = np.loadtxt('resultado.txt')

con = cn.connect(2037)

actions = ['left', 'right', 'jump']

def get_state_index(state):
    """Converts the binary string state to an index for the Q-table."""
    platform = int(state[2:7], 2)      # Platforms are 0-23
    direction = int(state[7:], 2)     # Directions are 00, 01, 10, 11
    return platform * 4 + direction

state = '0b0000000'
for i in range(100):
    action = actions[np.argmax(Q[get_state_index(state)])]# if random.random() <= 0.15 else random.choice(actions)
    state, reward = get_state_reward(con, action)
    print(get_state_index(state), Q[get_state_index(state)])

    




