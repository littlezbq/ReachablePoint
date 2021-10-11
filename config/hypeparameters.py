# Hypeparamter for training
import torch

class Hypeparam():
    def __init__(self):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.TOTAL_EPOCH = 500
        self.BATCH_SIZE = 64

        self.TRAIN_PERCENT = 0.8
        self.VAL_PERCENT = 0.1
        self.TEST_PERCENT = 1 - self.TRAIN_PERCENT - self.VAL_PERCENT

        self.INPUT_SIZE = 256
        self.INPUT_CHANNEL = 1
        self.OUTPUT_CLASS = 10

        self.LR = 3e-3
        self.STEP_SIZE = 20
        self.GAMMA = 0.1

    def showDevice(self):
        print(self.DEVICE)