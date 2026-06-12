from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def cifar_loaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError("dataset_name must be cifar10 or cifar100")
    dataset_cls = (
        datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.ToTensor()
    train_set = dataset_cls(
        data_root, train=True, transform=train_transform, download=download
    )
    test_set = dataset_cls(
        data_root, train=False, transform=test_transform, download=download
    )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    return (
        DataLoader(train_set, shuffle=True, **common),
        DataLoader(test_set, shuffle=False, **common),
    )

