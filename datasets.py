from typing import Iterable, Callable, Tuple
import os
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets
import torchvision.transforms as transforms


def _getRoot(rootDir: str = None) -> str:
    if rootDir is None:
        root = os.environ.get("DATASETS_ROOT")
        if root is None:
            root = "_data"
    else:
        root = rootDir
    return root


def getMNIST(
    batchSize: int,
    dataDir: str = None,
    trainTransforms: Iterable[Callable] = None,
    testTransforms: Iterable[Callable] = None,
) -> Tuple[TensorDataset, TensorDataset]:

    trainDataset = datasets.MNIST(
        _getRoot(dataDir),
        train=True,
        download=True,
        transform=trainTransforms,
        target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
    )
    testDataset = datasets.MNIST(
        _getRoot(dataDir),
        train=False,
        transform=testTransforms,
        target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
    )

    train_loader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )
    return train_loader, test_loader


mnistBaselineTrainTransforms = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)

mnistBaselineTestTransforms = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
    ]
)

mnistNoiseTrainTransforms = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.1 * (0.5 + 0.5 * torch.randn_like(x))),
    ]
)

mnistNoiseTestTransforms = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.1 * (0.5 + 0.5 * torch.randn_like(x))),
    ]
)


def mnistBaselineLoader(batchSize: int):
    return getMNIST(
        batchSize,
        trainTransforms=mnistBaselineTrainTransforms,
        testTransforms=mnistBaselineTestTransforms,
    )


def mnistNoiseLoader(batchSize: int):
    return getMNIST(
        batchSize,
        trainTransforms=mnistNoiseTrainTransforms,
        testTransforms=mnistNoiseTestTransforms,
    )


def getCifar10(
    batchSize: int,
    dataDir: str = None,
    trainTransforms: Iterable[Callable] = None,
    testTransforms: Iterable[Callable] = None,
) -> tuple:

    trainDataset = datasets.CIFAR10(
        _getRoot(dataDir),
        train=True,
        download=True,
        transform=trainTransforms,
        target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
    )

    testDataset = datasets.CIFAR10(
        _getRoot(dataDir),
        train=False,
        transform=testTransforms,
        target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
    )

    train_loader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )
    return train_loader, test_loader


cifarTrainTransforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

cifarTestTransforms = testTransforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def cifar10Loader(batchSize: int):
    return getCifar10(
        batchSize,
        trainTransforms=cifarTrainTransforms,
        testTransforms=cifarTestTransforms,
    )
