from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import fold
from captum.attr import IntegratedGradients


def attributions(
    model: nn.Module,
    samples: torch.Tensor,
    n_steps: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute attributions for given samples using integrated gradients

    Args:
        model (nn.Module): Trained model
        samples (torch.Tensor): Samples
        n_steps (int, optional): Discretization step for integrated gradient. Defaults to 50.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: model predictions, attributions
    """
    baseline = torch.zeros_like(samples)
    model.eval()
    with torch.no_grad():
        predictions = model(samples).squeeze().argmax(dim=1)

    integratedGrads = IntegratedGradients(model.forward, False)
    attrs = integratedGrads.attribute(samples, baseline, predictions, n_steps=n_steps)
    return predictions, attrs


def abc_metric(
    model: nn.Module, samples: torch.Tensor, targets: torch.Tensor, metricParams: dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the ABC score on a batch of samples. Absolute attributions are first divided by pixel values.
    These raw sensitivity images are then split into small patches. Each pixel is then assigned a probability
    (that sums up to 1 inside every patch). The patches are reassembled to generate a full probability image.
    The probability image gives Bernouilli parameters for conformity assessment over a given number of samples
    for abc metric assessment.
    The overall average abc score is returned, along with average scores for correctly classified samples and
    misclassified samples respectively.

    Args:
        model (nn.Module): Trained model
        samples (torch.Tensor): Samples
        targets (torch.Tensor): True targets
        metricParams (dict): ABC metric computation parameters

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: overall score, score for correct classifications,
        score for incorrect classifications
    """
    n_steps = metricParams["n_steps"]
    minPixelValue = metricParams["minPixelValue"]
    minProb = metricParams["minProb"]
    nConform = metricParams["nConform"]
    assert minPixelValue > 0
    assert 0 <= minProb <= 1

    batchSize, channels, height, width = samples.shape

    predictions, attrs = attributions(model, samples, n_steps=n_steps)

    filteredSamples = samples.clamp(
        min=minPixelValue
    )  # force samples to have minPixelValue as minimum
    ratio = (
        torch.abs(attrs / filteredSamples)
        .sum(dim=1)
        .reshape((batchSize, 1, height, width))
    )  # raw attribution images

    pSize = metricParams["patchSize"]  # patch size (square patches only)

    ratioPatches = ratio.unfold(2, pSize, pSize).unfold(
        3, pSize, pSize
    )  # splitting into patches
    ratioSum = (
        ratioPatches.sum((4, 5))
        .reshape((batchSize, 1, height // pSize, width // pSize, 1, 1))
        .expand_as(ratioPatches)
    )  # normalize by patch
    probsPatches = ratioPatches / ratioSum  # probability patches

    # reassemble patches into images
    probsPatches = (
        probsPatches.reshape(
            (batchSize, 1, height // pSize, width // pSize, pSize**2)
        )
        .permute(0, 1, 4, 2, 3)
        .squeeze(1)
        .reshape(batchSize, pSize**2, -1)
    )
    probs = fold(
        probsPatches, (height, width), kernel_size=pSize, stride=pSize
    )  # probability images

    scores = []
    scoresCorrect = []
    scoresIncorrect = []
    for sampleIdx, sample in enumerate(samples):
        fullMask = probs[sampleIdx].squeeze()
        conformBatchList = [
            # sample images using the probability images
            sample * ~(torch.bernoulli(fullMask.clamp_min(minProb)).bool())
            for _ in range(nConform)
        ]

        conformBatch = torch.stack(conformBatchList, dim=0)  # conform images batch

        with torch.no_grad():
            conformPredictions = model(conformBatch).argmax(
                dim=1, keepdim=True
            )  # compute new predictions
        predictedLabels = torch.ones_like(conformPredictions) * predictions[sampleIdx]
        score = (
            torch.eq(predictedLabels, conformPredictions).sum() / nConform
        )  # abc score
        scores.append(score)
        if predictions[sampleIdx] == targets[sampleIdx]:
            scoresCorrect.append(score)
        else:
            scoresIncorrect.append(score)

    return (
        torch.stack(scores, dim=0),
        torch.stack(scoresCorrect, dim=0) if len(scoresCorrect) > 0 else None,
        torch.stack(scoresIncorrect, dim=0) if len(scoresIncorrect) > 0 else None,
    )
