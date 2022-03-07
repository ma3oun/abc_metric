import logging
import os
import torch
from easydict import EasyDict

logger = logging.getLogger()


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def batch_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Batch accuracy in percent"""

    batch_size = target.size(0)
    predictedLabel = output.argmax(dim=1, keepdim=True)
    correct = predictedLabel.eq(target.view_as(predictedLabel)).sum()
    accuracy = 100.0 * correct / batch_size

    return accuracy


def checkSavedModel(params: EasyDict, model: torch.nn.Module) -> bool:
    """Check for a saved model

    Args:
        params (EasyDict): Training parameters
        model (torch.nn.Module): Constructed model

    Returns:
        bool: True if a saved model was found
    """
    modelName = f"{params.name}_{params.modelType}.pt"
    modelPath = os.path.join(params.savepoint, modelName)
    if not os.path.exists(modelPath):
        modelPath = os.path.join(os.getcwd(), modelPath)
    modelIsTrained = False
    if params.resume:
        if os.path.exists(modelPath):
            with open(modelPath, "rb") as f:
                model.load_state_dict(torch.load(f))
            modelIsTrained = True
        else:
            logger.warning("Saved model not found. Training from scratch.")
    return modelIsTrained


def saveModel(params: EasyDict, model: torch.nn.Module):
    """Save a trained model

    Args:
        params (EasyDict): Training parameters
        model (torch.nn.Module): Model to save
    """
    modelName = f"{params.name}_{params.modelType}.pt"
    if os.path.exists(params.savepoint):
        modelPath = os.path.join(params.savepoint, modelName)
    else:
        newDir = os.path.join(os.getcwd(), params.savepoint)
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        modelPath = os.path.join(newDir, modelName)
    torch.save(model.state_dict(), modelPath)
    return
