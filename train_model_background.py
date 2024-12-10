from minesweeper_learner import MineSweeperLearner
import sys
import getopt
import torch
import importlib.util
import importlib.machinery

def load_source(modname, filename):
    loader = importlib.machinery.SourceFileLoader(modname, filename)
    spec = importlib.util.spec_from_file_location(modname, filename, loader=loader)
    module = importlib.util.module_from_spec(spec)
    # Uncomment the following line to cache the module.
    # sys.modules[module.__name__] = module
    loader.exec_module(module)
    return module

def main(argv):
    option = ''
    modelChoice = ''
    nBatches = 1000
    nSamples = 1000
    epochsPerBatch = 1
    try:
        opts, args = getopt.getopt(argv, "ho:m:b:s:e:", ["option=", "model=", "batches=", "nSamples=", "epochsPerBatch="])
    except getopt.GetoptError:
        print('train_model_background.py -o <option> -m <model>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train_model_background.py -o <option> -m <model>')
            sys.exit()
        elif opt in ("-o", "--option"):
            option = arg
        elif opt in ("-m", "--model"):
            modelChoice = arg
        elif opt in ("-b", "--batches"):
            nBatches = int(arg)
        elif opt in ("-s", "--nSamples"):
            nSamples = int(arg)
        elif opt in ("-e", "--epochsPerBatch"):
            epochsPerBatch = int(arg)

    if option == "trainNew":
        modelSource = load_source(modelChoice, "modelCode/" + modelChoice + ".py")
        model = modelSource.model
    elif option == "continueTraining":
        model = torch.load("trainedModels/" + modelChoice + ".pt")
        model.eval()  # Ensure the model is in evaluation mode if continuing training.

    learner = MineSweeperLearner(modelChoice, model)
    learner.learnMineSweeper(nSamples, nBatches, epochsPerBatch, verbose=True)

    torch.save(learner.model.state_dict(), "trainedModels/" + modelChoice + ".pt")

if __name__ == "__main__":
   main(sys.argv[1:])
