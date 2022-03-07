import logging, coloredlogs
from typing import Callable
import random
import torch
from easydict import EasyDict
from learner import Learner
from evals import alphaBlendEval, rotationEval

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
coloredlogs.install(
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


def run(params: EasyDict, model: torch.nn.Module, datasetLoader: Callable):
    # Use CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not found!")

    model.cuda()

    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_loader, test_loader1 = datasetLoader(params.batchSize)

    if not params.modelIsTrained or not params.resume:
        logger.info(f"Training:")

        main_learner = Learner(
            params,
            model=model,
            trainloader=train_loader,
            testloader=test_loader1,
        )
        main_learner.learn()

        logger.info(f"Learning session complete")

    _, test_loader2 = datasetLoader(params.batchSize)

    if "alphaBlending" in params.evals:
        logger.info(f"Alpha blending evaluation using alpha={params.alpha}")
        alphaBlendEval(
            model, test_loader1, test_loader2, params.alpha, params.metricParams
        )
    if "rotation" in params.evals:
        logger.info(f"Rotation evaluation using angle={params.angle}")
        rotationEval(model, test_loader2, params.angle, params.metricParams)
