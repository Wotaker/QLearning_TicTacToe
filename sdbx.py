import random
import numpy as np
from enviroment import TicTacToeEnv
random.seed(67)

Q_PATH = "QualityFolder\\10e5_epochs.npy"
quality_function = np.load(Q_PATH)

print(quality_function[2, ])




