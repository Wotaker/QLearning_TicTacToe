import random
from enviroment import TicTacToeEnv
import numpy as np
import time
random.seed(42)

VERBOSE = False
QUALITY_FILE = "QualityFolder\\10e6_epochs.npy"

AI_SYMBOL, PLAYER_SYMBOL= 'X', 'O'
print(f"Ai has {AI_SYMBOL}, Player has {PLAYER_SYMBOL}")
myEnv = TicTacToeEnv(ai_symbol=AI_SYMBOL)

q_table = np.zeros([myEnv.observation_space.n, myEnv.action_space.n])

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.6
EPSILON = 0.1

EPOCHS = int(10e6)

print("=== Training has begun ===")
start = time.time()
for epoch in range(1, EPOCHS + 1):
    if VERBOSE: print("  Epoch ", epoch)

    state = myEnv.reset()
    reward = 0
    done = False
    start_symbol = random.choice(['X', 'O'])

    # If player has the first move, make it before the loop (Becouse int the loop we start from the AIs move)
    if start_symbol == PLAYER_SYMBOL:
        myEnv.player_step_random()
    while not done:
        possible_actions = myEnv.possibleActions()
        if random.uniform(0, 1) < EPSILON:  # Explore action space
            action = random.choice(possible_actions)
        else:   # Exploit learned values
            action = possible_actions[0]
            for a in possible_actions[1:]:
                if q_table[state, a - 1] > q_table[state, action - 1]:
                    action = a
        
        next_state, reward, done, info = myEnv.ai_step(action, player_step=True)
        old_value = q_table[state, action - 1]

        possible_actions = myEnv.possibleActions()
        if not possible_actions and not done:
            print("Warning! There are no actions left, but the game is sill on!")

        if VERBOSE and done:
            myEnv.render()
            print(f"Start Symbol: {start_symbol}, State: {next_state}, Reward: {reward}")

        # estimating the next best action quality
        if not possible_actions:    # No possible actions, we are done
            next_max = 0
        else:                       # Some maximal quality can be estimated
            next_max = q_table[next_state, possible_actions[0] - 1]
            for a in possible_actions[1:]:
                if q_table[next_state, a - 1] > next_max:
                    next_max = q_table[next_state, a - 1]
        
        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        q_table[state, action - 1] = new_value

        state = next_state
    
    if epoch % 50000 == 0:
        print(f"Epoch {epoch}")

end = time.time()
np.save('QualityFunction.npy', q_table)
print(f"=== Training has been compleated in {end - start}s ===")

