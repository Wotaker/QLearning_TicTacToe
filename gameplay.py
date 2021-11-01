from enviroment import TicTacToeEnv
import numpy as np

Q_PATH = "QualityFolder\\10e5_epochs.npy"

def gameplay(q_function_path, player_symbol='O', starting_symbol='X'):
    """
    Runs the game
    ATENTION! Player symbol must only be 'O', becouse AI was trained while beeing an 'X'
    TODO change this restriction
    """
    quality_function = np.load(q_function_path)

    def getBestAction(current_state, possible_actions):
        best_action = possible_actions[0]
        for a in possible_actions[1:]:
            if quality_function[current_state, a - 1] > quality_function[current_state, best_action - 1]:
                best_action = a
        return best_action

    ai_symbol = 'X'
    if player_symbol == 'X':
        ai_symbol = 'O'
    env = TicTacToeEnv(ai_symbol)
    state = env.reset()
    info = f"""
    === This is a TicTackToe game! ====================================
      You are plaing against a Q-learning model.
      Your symbol is {player_symbol}, so the AIs symbol is {ai_symbol}

    """
    print(info)
    print("  Game state after initialization:")
    env.render()

    done = False
    winner = None
    if starting_symbol == ai_symbol:
        possible_actions = env.possibleActions()
        ai_action = getBestAction(state, possible_actions)
        state, _, done, _ = env.ai_step(ai_action)
        print("AI action: ", ai_action)
        print(f"  Game state after AIs action:")
        env.render()
    
    while not done:
        # players move
        player_action = 0
        possible_actions = env.possibleActions()
        while not (player_action in possible_actions):
            player_action = int(input(
                f"Please take one of possible actions: {possible_actions}\nYour action: "
                ))
        print(f"Your action is {player_action} -> {env.actionToCoords(player_action)}")

        env.player_step_action(player_action)
        winner, done = env.gameState()
        state = env.getCurrentState()

        print(f"  Game state after your action {player_action}:")
        env.render()
        
        if not done:
            possible_actions = env.possibleActions()
            ai_action = getBestAction(state, possible_actions)
            state, _, done, _ = env.ai_step(ai_action)
            winner, _ = env.gameState()
            print("AI action: ", ai_action)
            print(f"  Game state after AIs action:")
            env.render()
    
    # We are done, game is over
    winner = env.symbols[winner]
    if winner == player_symbol:
        print(f"Congratulations!!! You have won the game!")
    elif winner == ai_symbol:
        print(f"You have lost with a machine dummy! Try again later.")
    else:
        print(f"The game has ended with a draw, nothing new in TicTacToe...")


gameplay(Q_PATH, player_symbol='O', starting_symbol='O')
