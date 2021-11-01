import random
import numpy as np
import gym
from gym import spaces


class TicTacToeEnv(gym.Env):
  """
  Custom Environment that follows gym interface. This is a basic Tic Tac Toe enviroment.
  """

  SIZE = 3

  nums = {'.': 0, 'X': 1, 'O': 2}
  symbols = {0: '.', 1: 'X', 2: 'O'}


  def __init__(self, ai_symbol='X'):
    super(TicTacToeEnv, self).__init__()
    self.action_space = spaces.Discrete(self.SIZE * self.SIZE)
    self.observation_space = spaces.Discrete(3 ** (self.SIZE * self.SIZE))

    # actions count from 1 to 9, starting from top left corner
    self.actions = set([a for a in range(1, self.SIZE * self.SIZE + 1)])
    self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
    self.ai_symbol = ai_symbol
    self.player_symbol = 'O'
    if self.ai_symbol == 'O':
      self.player_symbol = 'X'
    self.moves_done = 0
        
    return None

  @staticmethod
  def actionToCoords(a, size=3):
    """
    returns action a translated to (x, y) coordinates on the board (starting from (0, 0))
    """
    return ((a + size - 1) // size) - 1, (a - 1) % size

  
  @staticmethod
  def coordsToAction(coords, size=3):
    return (size * coords[0]) + coords[1] + 1
  

  def gameState(self):
    """
    Returns tuple with the winner/loser/draw symbol if ended or draw symbol else, 
    and information if the game has ended
    """

    nill_tile = 0

    def horizontal():
      for row in range(self.SIZE):
        suspected = self.board[row][0]
        if suspected != nill_tile:
          for col in range(1, self.SIZE):
            if self.board[row][col] != suspected:
              suspected = nill_tile
              break
        if suspected != nill_tile:
          return suspected
      return nill_tile
    
    def vertical():
      for col in range(self.SIZE):
        suspected = self.board[0][col]
        if suspected != nill_tile:
          for row in range(1, self.SIZE):
            if self.board[row][col] != suspected:
              suspected = nill_tile
              break
        if suspected != nill_tile:
          return suspected
      return nill_tile
    
    def diagonal():
      suspected = self.board[0][0]
      if suspected != nill_tile:
        for i in range(1, self.SIZE):
          if self.board[i][i] != suspected:
            suspected = nill_tile
            break
      if suspected != nill_tile:
          return suspected

      suspected = self.board[self.SIZE - 1][0]
      if suspected != nill_tile:
        for i in range(1, self.SIZE):
          if self.board[self.SIZE - 1 - i][i] != suspected:
            suspected = nill_tile
            break
      if suspected != nill_tile:
          return suspected
      return nill_tile
    
    suspected = horizontal()
    if suspected != nill_tile:
      return suspected, True
    suspected = vertical()
    if suspected != nill_tile:
      return suspected, True
    suspected = diagonal()
    if suspected != nill_tile:
      return suspected, True
    
    full = True
    for row in range(self.SIZE):
      for col in range(self.SIZE):
        if self.board[row][col] == 0:
          full = False

    return nill_tile, full


  def reset(self):
    """
    Resets the game
    :return: board
    """
    self.board = np.zeros((self.SIZE, self.SIZE))
    self.moves_done = 0
    return self._encodeState()


  def ai_step(self, action, player_step=False):
    """
    Makes a step according to action and current board state

    Reward for:
      - losing move -> -5
      - illegal move -> -4
      - neutral move -> 0
      - draw move -> 3
      - winning move -> 5

    :returns: encoded state, rewars, if done, info dictionary
    """
    reward_dict = {"losing": -5, "illegal": -4, "neutral": 0, "draw": 3, "winning": 5}
    reward = None
    winner = None
    info = {'prob': 1.0}
    done = False
      

    # check if game is still on
    winner, done = self.gameState()
    if done:
      if winner == self.nums['.']:
        return self._encodeState(), reward_dict['draw'], True, info
      elif winner == self.nums[self.player_symbol]:
        return self._encodeState(), reward_dict['losing'], True, info

    # Validate the action
    if action not in self.actions:
      raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    # translating action to coords
    action_coords = self.actionToCoords(action, self.SIZE)
    
    # checkking if not placed on already filled tile, if so, negative reward, and log in info
    if self.board[action_coords[0]][action_coords[1]] != self.nums['.']:
      reward = reward_dict['illegal']
    # otherwise, we take the action (make the move)
    else:
      self.board[action_coords[0]][action_coords[1]] = self.nums[self.ai_symbol]
      # checkking game state after the move:
      winner, done = self.gameState()
      if not done and player_step is True:    # if players move as well, and we are not done, we recalculate
        self.player_step_random()  # the game State after the players move
        winner, done = self.gameState()
      if done:
        if winner == self.nums['.']:
          return self._encodeState(), reward_dict['draw'], True, info
        elif winner == self.nums[self.ai_symbol]:
          return self._encodeState(), reward_dict['winning'], True, info
        else:
          return self._encodeState(), reward_dict['losing'], True, info
      else:
        reward = reward_dict['neutral']

    # returns the appropiate reward
    return self._encodeState(), reward, False, info
  

  def possibleActions(self):
    """
    Returns the list of possible actions (moves)
    """
    possible_actions = []
    for row in range(self.SIZE):
      for col in range(self.SIZE):
        if self.board[row][col] == self.nums['.']:
          possible_actions.append(self.coordsToAction((row, col), self.SIZE))
    return possible_actions


  def player_step_random(self):
    """
    Takes the move in one of possible tiles.
    Returns the possible actions after the move.
    """  
    possible_actions = self.possibleActions()
    players_move = self.actionToCoords(random.choice(possible_actions))
    self.board[players_move[0]][players_move[1]] = self.nums[self.player_symbol]
  

  def player_step_action(self, action):
    if not (action in self.possibleActions()):
      print(
        f"Error! This action can not be done. Possible acions were {self.possibleActions()} and yours was {action}"
        )
      exit(-1)
    players_move = self.actionToCoords(action)
    self.board[players_move[0]][players_move[1]] = self.nums[self.player_symbol]


  def _encodeState(self):
    """
    Encodes the board state into an integer state.
    Private method
    """
    state = 0
    power = 0
    for row in range(self.SIZE):
      for col in range(self.SIZE):
        state += int((3**power) * self.board[row][col])
        power += 1
    
    return state

  
  def getCurrentState(self):
    return self._encodeState()


  def render(self, mode='console'):
    """
    Renders the board according to a mode.
    If mode = console, than prints the board in ASCII
    """

    if mode != 'console':
      raise NotImplementedError()

    for col in range(self.SIZE):
      print("+---", end="")
    print("+")
    for row in range(self.SIZE):
      for col in range(self.SIZE):
        print(f"| {self.symbols[self.board[row][col]]} ", end="")
      print("|")
      for col in range(self.SIZE):
        print("+---", end="")
      print("+")
    return


