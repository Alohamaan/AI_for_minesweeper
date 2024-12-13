import os
import numpy as np
import train_model_background
import importlib.util
import importlib.machinery

#Prompt user to specify the model they want to use
#get raw model code

def load_source(modname, filename):
    loader = importlib.machinery.SourceFileLoader(modname, filename)
    spec = importlib.util.spec_from_file_location(modname, filename, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


models = os.listdir("../modelCode")
models = [i.replace(".py","") for i in models if not ".pyc" in i and i[0] != '.']
models = np.sort(models)

#and pre-trained models
preTrainedModels = os.listdir("../trainedModels")
preTrainedModels = np.sort(preTrainedModels)

toDo = int(input("What do you want to do? \n1. Train a new model from scratch \n2. Keep training a pre-trained model\n"))

modelChoice = ''
match toDo:
    case 1:
        prompt = "Choose which model to train (from 'modelCode' folder): \n"
        for i in range(len(models)):
            prompt += str(i+1) +  ". " + models[i] + '\n'
        modelChoice = int(input(prompt))
        modelChoice = models[modelChoice-1]
    case 2:
        prompt = "Choose which model to continue training (from 'trainedModels' folder): \n"
        for i in range(len(preTrainedModels)):
            prompt += str(i+1) +  ". " + preTrainedModels[i] + '\n'
        modelChoiceInd = int(input(prompt))
        modelChoice = preTrainedModels[modelChoiceInd-1]

#get batch info
samples = int(input("How many samples per batch? "))
nBatches = int(input("How many batches? "))
nEpochsPerBatch = int(input("How many training epochs on each batch? "))

print(modelChoice)
print(nBatches)
print(samples)
print(nEpochsPerBatch)

#launch background process
if toDo == 1:
    args = [
        "train_model_background.py",  # This is typically the script name
        "-o", "trainNew",
        "-m", modelChoice,
        "-b", str(nBatches),
        "-s", str(samples),
        "-e", str(nEpochsPerBatch)
    ]

elif toDo == 2:
    args = [
        "train_model_background.py",  # This is typically the script name
        "-o", "continueTraining",
        "-m", modelChoice,
        "-b", str(nBatches),
        "-s", str(samples),
        "-e", str(nEpochsPerBatch)
    ]
    train_model_background.main(args)

print("Model training output is being written to log/" + modelChoice + ".out")
print("Model will be saved every 100 batches to trainedModels/" + modelChoice)
