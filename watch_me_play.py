import os
from final_nn_for_minesweeper.model_training.minesweeper_learner import MineSweeperLearner
import torch
import numpy as np
from model_code import model1
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

dim1 = 16
dim2 = 30
model = model1.CustomModel(dim1, dim2)
model.load_state_dict(torch.load(f'trainedModels/{modelChoice}'), strict=False)
model.eval()

learner = MineSweeperLearner(modelChoice, model)
learner.watchMePlay()
