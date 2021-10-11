import glob
import os

import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from utils import draw_umich_gaussian
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split


class KeyPointDatasets(Dataset):
    def __init__(self, root_dir,  transforms=None):
        super(KeyPointDatasets, self).__init__()

        self.down_ratio = 1
        # self.img_w = 480 // self.down_ratio
        # self.img_h = 360 // self.down_ratio
        self.root_dir = root_dir
        self.img_w = 256 // self.down_ratio
        self.img_h = 256 // self.down_ratio

        self.img_path = os.path.join(self.root_dir, "images/")

        self.img_list = glob.glob(os.path.join(self.img_path, "*.tif"))
        self.label_list = [item.replace("images", "labels") for item in self.img_list]

        # 拆分训练集、验证集和测试集
        # img_train_list, img_test_list = train_test_split(self.img_list,test_size=param.TEST_PERCENT,
        #                                        train_size=param.TRAIN_PERCENT,random_state=47,shuffle=False)
        #
        # img_train_list_, img_val_list = train_test_split(self.img_list,test_size=param.VAL_PERCENT,
        #                                        random_state=47,shuffle=False)


        if transforms is not None:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]

        img = img_path.replace('\\', '/')
        raw_img = Image.open(img_path)
        trans_img = np.float32(np.array(raw_img) / 8000) #归一化

        trans_label = np.array(Image.open(label_path))
        if self.transforms:
            trans_img = self.transforms(trans_img)
            trans_label = self.transforms(trans_label)

        # with open(txt, "r") as f:
        #     for i, line in enumerate(f):
        #         if i == 0:
        #             # 第一行
        #             num_point = int(line.strip())
        #             heatmap = np.zeros((self.img_h, self.img_w))
        #         else:
        #             x1, y1 = [(t.strip()) for t in line.split()]
        #             # range from 0 to 1
        #             x1, y1 = float(x1), float(y1)
        #
        #             cx, cy = x1 * self.img_w, y1 * self.img_h
        #
        #             draw_umich_gaussian(heatmap, (cx, cy), 30, 2)

        return trans_img, trans_label

    def __len__(self):
        return len(self.img_list)

    # @staticmethod
    # def collect_fn(batch):
    #     imgs, labels = zip(*batch)
    #     return torch.stack(imgs, 0), torch.stack(labels, 0)


if __name__ == "__main__":
    trans = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((360, 480)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    kp_datasets = KeyPointDatasets(
        root_dir="datasets/Qingmuchuan/", transforms=trans)

    # for i in range(len(kp_datasets)):
    # print(kp_datasets[i][0].shape, kp_datasets[i][1])

    data_loader = DataLoader(kp_datasets, num_workers=0, batch_size=4, shuffle=True)

    for data, label in data_loader:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(data[0][0])
        ax[1].imshow(label[0][0])
        print(data.dtype, label.shape)
        plt.show()
