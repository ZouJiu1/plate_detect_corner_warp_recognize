import sys
import os
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
sys.path.append(os.getcwd())
import torchvision.transforms as transforms
import torch

pwd = os.path.abspath('./')

# 训练数据的变换
train_data_transforms = transforms.Compose([
    transforms.Resize([60, 190]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# 测试数据的变换
test_data_transforms = transforms.Compose([
    transforms.Resize([60, 190]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

class TrainDataset(Dataset):
    def __init__(self, dir, transform=None):
        super(TrainDataset, self).__init__()
        self.imgpath = [os.path.join(dir, i) for i in os.listdir(dir) if 'jpg' in i]
        self.transform = transform

    def __len__(self):
        return len(self.imgpath)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgpath[idx])
        txtfile = self.imgpath[idx].replace('rectangle', 'label').replace('.jpg','.txt').replace('.png', '.txt')
        with open(txtfile, 'r') as obj:
            coords = obj.readline().strip()
            coords = [float(a) for a in coords.split(' ')]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(coords,dtype=torch.float32)

class TestDataset(Dataset):
    def __init__(self, dir, transform=None):
        super(TestDataset, self).__init__()
        self.imgpath = [os.path.join(dir, i) for i in os.listdir(dir) if 'jpg' in i]
        self.transform = transform

    def __len__(self):
        return len(self.imgpath)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgpath[idx])
        txtfile = self.imgpath[idx].replace('.jpg','.txt').replace('.png', '.txt')
        with open(txtfile, 'r') as obj:
            coords = obj.readline().strip()
            coords = [float(a) for a in coords.split(' ')]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)
        return img,torch.tensor(coords,dtype=torch.float32)

# 训练数据生成器
train_dataloader = torch.utils.data.DataLoader(
    dataset = TrainDataset(dir=os.path.join(pwd, 'data', 'corner','rectangle'),\
                           transform=train_data_transforms),
    batch_size = 16,
    num_workers = 9,
    shuffle = False
)

# 测试数据生成器
test_dataloader = torch.utils.data.DataLoader(
    dataset=TestDataset(dir=os.path.join(pwd, 'data', 'corner','test'),\
                         transform=train_data_transforms),
    batch_size=16,
    num_workers=9,
    shuffle=True
)

if __name__ =='__main__':
    for i, img, coord in enumerate(test_dataloader):
        i=1