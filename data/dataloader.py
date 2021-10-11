import torch
from torch.utils.data import DataLoader

from config import Hypeparam
from datasets import KeyPointDatasets
from torchvision import transforms as T

param = Hypeparam()


def make_dataLoader(root_dir):
    transforms_all = T.Compose([
        T.ToPILImage(),
        T.Resize((param.INPUT_SIZE, param.INPUT_SIZE)),
        T.ToTensor(),
        # T.Normalize(mean=[0.4372, 0.4372, 0.4373],
        #             std=[0.2479, 0.2475, 0.2485])
    ])

    datasets = KeyPointDatasets(root_dir=root_dir, transforms=transforms_all)
    train_size = int(param.TRAIN_PERCENT * len(datasets))
    val_size = int(param.VAL_PERCENT * len(datasets))
    test_size = len(datasets) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(datasets,
                                                                             [train_size, val_size, test_size],
                                                                             generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=param.BATCH_SIZE)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=param.BATCH_SIZE)
    test_loader = DataLoader(val_dataset, shuffle=False, batch_size=1)

    return train_loader, val_loader, test_loader
