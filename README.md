# QLearning_TicTacToe
In this project I use the Quality Reinforcement Learning to train an agent playing simple TicTacToe. The enviroment is OpenAI gym style.
The main rule behind this training is:

<img src="https://www.codecogs.com/eqnedit.php?latex=Q(state,&space;action)\leftarrow(1-\alpha)Q(state,&space;action)&space;&plus;&space;\alpha&space;(reward&space;&plus;&space;\gamma&space;\underset{a\in&space;actions}{max}Q(state_{next},&space;a))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(state,&space;action)\leftarrow(1-\alpha)Q(state,&space;action)&space;&plus;&space;\alpha&space;(reward&space;&plus;&space;\gamma&space;\underset{a\in&space;actions}{max}Q(state_{next},&space;a))" title="Q(state, action)\leftarrow(1-\alpha)Q(state, action) + \alpha (reward + \gamma \underset{a\in actions}{max}Q(state_{next}, a))" /></a>

## Libraries
* gym
* numpy

## Run the game
### Training new model
* After specifying the right hyperparameters and the number of training epochs (one epoch is one game) just run the *training.py* file. The model (numpy 2d array)
will be saved automaticlly into the folder *QualityFolder*
* In order to just run the game you do NOT NEED to train new model
### Playing against the trained model
* Just run the *gameplay.py* file in your python interpreter
* Follow the command line instructions
* You can choose which model to play against by specifying the right path in the *Q_PATH* variable. The models (Quality tables) are saved in the *QualityFolder*
