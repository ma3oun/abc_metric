import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from abc_utils import abc_metric
from train_utils import AverageMeter, batch_accuracy
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torchvision.transforms.functional import rotate


def rotationEval(
    model: nn.Module,
    dataloader: TensorDataset,
    rotationAngle: float,
    metricParams: dict,
):
    model.eval()
    scores = AverageMeter()
    scoresCorrect = AverageMeter()
    scoresIncorrect = AverageMeter()
    accuracies = AverageMeter()
    with logging_redirect_tqdm():
        with tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Confidence: ",
        ) as bar:
            for _, (x, y) in bar:
                batch_size = x.size(0)
                x = x.cuda()
                y = y.cuda()

                rotatedX = rotate(x, rotationAngle)  # alpha blend the two images
                with torch.no_grad():
                    outputs = model(rotatedX).squeeze()
                accuracy = batch_accuracy(outputs, y)
                score, scoreCorrect, scoreIncorrect = abc_metric(
                    model, rotatedX, y, metricParams
                )

                scoreAvg = 100.0 * score.mean()
                if not scoreCorrect is None:
                    scoreCorrectAvg = 100.0 * scoreCorrect.mean()
                    scoresCorrect.update(scoreCorrectAvg.item(), scoreCorrect.shape[0])
                if not scoreIncorrect is None:
                    scoreIncorrectAvg = 100.0 * scoreIncorrect.mean()
                    scoresIncorrect.update(
                        scoreIncorrectAvg.item(), scoreIncorrect.shape[0]
                    )

                accuracies.update(accuracy.item(), batch_size)
                scores.update(scoreAvg.item(), batch_size)

                bar.set_description(
                    f"Confidence: Acc: {accuracies.avg: .4f} | Score: {scores.avg: .4f} | (correct): {scoresCorrect.avg: .4f} | (incorrect): {scoresIncorrect.avg: .4f}"
                )
    return


def alphaBlendEval(
    model: nn.Module,
    dataloader1: TensorDataset,
    dataloader2: TensorDataset,
    alpha: float,
    metricParams: dict,
):
    """Compute ABC scores by alpha belnding data from 2 dataloaders

    Args:
        model (nn.Module): Trained model
        dataloader1 (TensorDataset): First dataloader
        dataloader2 (TensorDataset): Second dataloader
        alpha (float): Alpha blending factor
        metricParams (dict): ABC score computation parameters
    """
    model.eval()
    scores = AverageMeter()
    scoresCorrect = AverageMeter()
    scoresIncorrect = AverageMeter()
    accuracies = AverageMeter()
    fullDataloader = zip(dataloader1, dataloader2)
    with logging_redirect_tqdm():
        with tqdm(
            enumerate(fullDataloader),
            total=len(dataloader1),
            desc="Confidence: ",
        ) as bar:
            for _, ((x1, y1), (x2, y2)) in bar:
                batch_size = x1.size(0)
                x1 = x1.cuda()
                y1 = y1.cuda()
                x2 = x2.cuda()
                y2 = y2.cuda()

                blendX = (1 - alpha) * x1 + alpha * x2  # alpha blend the two images
                blendY = y1 if alpha < 0.5 else y2  # get the most likely label
                with torch.no_grad():
                    outputs = model(blendX).squeeze()
                accuracy = batch_accuracy(outputs, blendY)
                score, scoreCorrect, scoreIncorrect = abc_metric(
                    model, blendX, blendY, metricParams
                )

                scoreAvg = 100.0 * score.mean()
                if not scoreCorrect is None:
                    scoreCorrectAvg = 100.0 * scoreCorrect.mean()
                    scoresCorrect.update(scoreCorrectAvg.item(), scoreCorrect.shape[0])
                if not scoreIncorrect is None:
                    scoreIncorrectAvg = 100.0 * scoreIncorrect.mean()
                    scoresIncorrect.update(
                        scoreIncorrectAvg.item(), scoreIncorrect.shape[0]
                    )

                accuracies.update(accuracy.item(), batch_size)
                scores.update(scoreAvg.item(), batch_size)

                bar.set_description(
                    f"Confidence: Acc: {accuracies.avg: .4f} | Score: {scores.avg: .4f} | (correct): {scoresCorrect.avg: .4f} | (incorrect): {scoresIncorrect.avg: .4f}"
                )
    return
