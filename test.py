from model_code import model1
from model_training.minesweeper_learner import MineSweeperLearner
import torch


nSamples = 128
nBatches = 3000
epochsPerBatch = 5
if __name__ == '__main__':
    model = model1.model
    learner = MineSweeperLearner('model1.py', model)
    learner.learn_mine_sweeper(nSamples, nBatches, epochsPerBatch, verbose=True)

    torch.save(learner.model.state_dict(), "pretrained_models/" + 'model1' + ".pt")
