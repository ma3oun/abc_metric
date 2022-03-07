from typing import List
import logging
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from easydict import EasyDict

from train_utils import batch_accuracy, AverageMeter, saveModel

logger = logging.getLogger("learner")


class Learner:
    def __init__(
        self,
        params: EasyDict,
        model: nn.Module,
        trainloader: TensorDataset,
        testloader: TensorDataset,
    ) -> None:
        logger.name = params.name
        self.model = model
        self.epochs = params.epochs
        self.trainloader = trainloader
        self.testloader = testloader
        self.schedule = params.schedule
        self.lr = params.lr
        self.gamma = params.gamma
        self.params = params

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )

        self.model.cuda()

    def train(self):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        lossFn = nn.CrossEntropyLoss()

        with logging_redirect_tqdm():
            with tqdm(
                enumerate(self.trainloader),
                total=len(self.trainloader),
                desc="Train: ",
            ) as bar:
                for batch_idx, (inputs, targets) in bar:
                    # measure data loading time
                    data_time.update(time.time() - end)

                    inputs, targets = (
                        inputs.cuda(),
                        targets.cuda(),
                    )

                    # compute output
                    outputs = self.model(inputs).squeeze()
                    loss = lossFn(outputs, targets)
                    accuracy = batch_accuracy(outputs, targets)
                    batch_size = inputs.size(0)
                    losses.update(loss.item(), batch_size)
                    top1.update(accuracy.item(), batch_size)

                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # plot progress
                    bar.set_description(
                        f"Train: Loss: {losses.avg:.4f} | Acc: {top1.avg: .4f}"
                    )

        self.train_loss, self.train_acc = losses.avg, top1.avg

    def test(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        lossFn = nn.CrossEntropyLoss()

        self.model.eval()

        end = time.time()
        with logging_redirect_tqdm():
            with tqdm(
                enumerate(self.testloader),
                total=len(self.testloader),
                desc="Test: ",
            ) as bar:
                for batch_idx, (inputs, targets) in bar:
                    # measure data loading time
                    data_time.update(time.time() - end)

                    inputs, targets = (
                        inputs.cuda(),
                        targets.cuda(),
                    )

                    outputs = self.model(inputs).squeeze()

                    loss = lossFn(outputs, targets)

                    accuracy = batch_accuracy(outputs, targets)
                    batch_size = inputs.size(0)
                    losses.update(loss.item(), batch_size)
                    top1.update(accuracy.item(), batch_size)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # plot progress
                    bar.set_description(
                        f"Test: Loss: {losses.avg:.4f} | Acc: {top1.avg: .4f}"
                    )

        self.test_loss = losses.avg
        self.test_acc = top1.avg

    def adjust_learning_rate(self, epoch):
        if epoch in self.schedule:
            self.lr *= self.gamma
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

    def learn(self):
        for epoch in range(self.epochs):
            self.adjust_learning_rate(epoch)

            logger.info(f"\nEpoch: [{epoch + 1} | {self.epochs}] LR: {self.lr}")
            # with torch.autograd.set_detect_anomaly(True): # debug only
            self.train()
            self.test()
        saveModel(self.params, self.model)
