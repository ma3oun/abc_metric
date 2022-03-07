from argparse import ArgumentError
import logging

from runner import run
from models import ResNet
from datasets import cifar10Loader
from easydict import EasyDict

from train_utils import checkSavedModel

logger = logging.getLogger()

params = EasyDict()
params.savepoint = "cifar"
params.modelType = "ResNet"
params.resume = True
params.name = "cifar10"
params.epochs = 50
params.lr = 0.001
params.batchSize = 16
params.schedule = [20, 30, 40, 45]  # learning rate schedule
params.gamma = 0.5
params.alpha = 0.01
params.angle = 0
params.evals = ["alphaBlending", "rotation"]
params.metricParams = {
    "n_steps": 50,  # integrated gradients discretization steps
    "minPixelValue": 1e-5,  # minimal pixel value
    "minProb": 0.0,  # minimal pixel switching probability
    "patchSize": 4,  # patch size
    "nConform": 50,  # conformance samples to generate
}


if __name__ == "__main__":
    logger.info(params)
    if params.modelType == "ResNet":
        model = ResNet(3)
    else:
        raise ArgumentError(f"Invalid model type: {params.modelType}")

    params.modelIsTrained = checkSavedModel(params, model)

    run(params, model, cifar10Loader)
