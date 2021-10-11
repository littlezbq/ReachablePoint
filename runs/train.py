import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data import make_dataLoader
from data.datasets import KeyPointDatasets
from modeling import KeyPointModel
from config.hypeparameters import Hypeparam


def train(model, epoch, dataloader, optimizer, criterion, scheduler, param):
    model.train()
    for itr, (image, label) in enumerate(dataloader):
        label = label.to(param.DEVICE)
        image = image.to(param.DEVICE)

        bs = image.shape[0]

        output = model(image)

        label = label.float()

        l = criterion(output, label)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if itr % 2 == 0:
            print("epoch:%2d|step:%04d|loss:%.6f" %
                  (epoch, itr, l.item() / bs))
            # vis.plot_many_stack({"train_loss": loss.item()/bs})
    scheduler.step()


def test(model, epoch, dataloader, criterion, param):
    model.eval()
    sum_loss = 0.
    n_sample = 0
    for itr, (image, label) in enumerate(dataloader):
        label = label.to(param.DEVICE)
        image = image.to(param.DEVICE)

        output = model(image)
        label = label.float()

        l = criterion(output, label)

        sum_loss += l.item()
        n_sample += image.shape[0]

    print("TEST: epoch:%02d-->loss:%.6f" % (epoch, sum_loss / n_sample))
    # if epoch > 1:
    #     vis.plot_many_stack({"test_loss": sum_loss/n_sample})
    return sum_loss / n_sample


if __name__ == "__main__":
    param = Hypeparam()
    data_root = r"F:\Projects\Pytorch\Gxy-Template\data\datasets\Qingmuchuan"
    train_loader, val_loader, test_loader = make_dataLoader(data_root)

    model = KeyPointModel().to(param.DEVICE)
    param.showDevice()

    optimizer = torch.optim.Adam(model.parameters(), lr=param.LR)
    criterion = torch.nn.MSELoss()  # compute_loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=param.STEP_SIZE, gamma=param.GAMMA)

    for epoch in range(param.TOTAL_EPOCH):
        train(model, epoch, train_loader, optimizer, criterion, scheduler, param)
        loss = test(model, epoch, val_loader, criterion,param)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "logs/epoch_%d_%.3f.pt" % (epoch, loss * 10000))
