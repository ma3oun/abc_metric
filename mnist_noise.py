from argparse import ArgumentError
import logging
from runner import run
from models import MLP, ResNet
from datasets import mnistNoiseLoader
from easydict import EasyDict

from train_utils import checkSavedModel

logger = logging.getLogger()

params = EasyDict()
params.savepoint = "noise"
params.modelType = "ResNet"
params.resume = True
params.name = "mnist"
params.epochs = 10
params.lr = 0.001
params.batchSize = 16  # 128
params.schedule = [6, 8, 16]
params.gamma = 0.5
params.alpha = 0.49
params.eval = ["alphaBlending"]
params.metricParams = {
    "n_steps": 50,
    "minPixelValue": 1e-5,
    "minProb": 0.0,
    "patchSize": 2,
    "nConform": 50,
}


if __name__ == "__main__":
    logger.info(params)
    if params.modelType == "MLP":
        model = MLP()
    elif params.modelType == "ResNet":
        model = ResNet(1)
    else:
        raise ArgumentError(f"Invalid model type: {params.modelType}")

    params.modelIsTrained = checkSavedModel(params, model)

    run(params, model, mnistNoiseLoader)
