from modelCode import model1
from minesweeper_learner import MineSweeperLearner
import torch


nSamples = 128
nBatches = 3000
epochsPerBatch = 5
if __name__ == '__main__':
    model = model1.model
    learner = MineSweeperLearner('model1.py', model)
    learner.learnMineSweeper(nSamples, nBatches, epochsPerBatch, verbose=True)

    torch.save(learner.model.state_dict(), "trainedModels/" + 'model1' + ".pt")
