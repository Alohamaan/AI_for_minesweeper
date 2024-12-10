import os
from minesweeper_learner import MineSweeperLearner
import torch
import numpy as np
from warnings import filterwarnings
filterwarnings("ignore")

np.set_printoptions(linewidth = 1000, precision=3, suppress=True)

#Prompt user to specify the model they want to watch play
preTrainedModels = os.listdir("trainedModels")
preTrainedModels = [i for i in preTrainedModels]
preTrainedModels = np.sort(preTrainedModels)
prompt = "Which model do you want to watch play? \n"
for i in range(len(preTrainedModels)):
    prompt += str(i + 1) + ". " + preTrainedModels[i] + '\n'
modelChoice = int(input(prompt))
modelChoice = preTrainedModels[modelChoice - 1]

model = torch.load_model(modelChoice)

learner = MineSweeperLearner(modelChoice, model)
learner.watchMePlay()