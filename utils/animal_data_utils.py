from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from utils.ws_augmentation import TransformFixMatchMedium
import os


class Animal10N_dataset(Dataset):

    def __init__(self, root_dir, mode):

        self.root_dir = root_dir
        self.mode = mode
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_fixmatch = TransformFixMatchMedium((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214), 3, 10)

        self.train_dir = root_dir + '/training/'
        self.test_dir = root_dir + '/testing/'
        train_imgs = os.listdir(self.train_dir)
        test_imgs = os.listdir(self.test_dir)
        self.train_imgs = []
        self.test_imgs = []
        self.train_targets = []
        self.test_targets = []

        for img in train_imgs:
            label = int(img[0])
            self.train_imgs.append([img, int(img[0])])
            self.train_targets.append(label)
        for img in test_imgs:
            label = int(img[0])
            self.test_imgs.append([img, int(img[0])])
            self.test_targets.append(label)

        # Set targets attribute based on mode
        if mode == 'train':
            self.targets = self.train_targets
        elif mode == 'test':
            self.targets = self.test_targets

    def __getitem__(self, index):

        if self.mode == 'train':
            ind = index
            img_id, target = self.train_imgs[index]
            img_path = self.train_dir + img_id
            image = Image.open(img_path).convert('RGB')
            img_w = self.transform_fixmatch.weak(image)
            img_s = self.transform_fixmatch.strong(image)
            return img_w, img_s, target, ind

        elif self.mode == 'test':
            ind = index
            img_id, target = self.test_imgs[index]
            img_path = self.test_dir + img_id
            image = Image.open(img_path).convert('RGB')
            img = self.transform_test(image)
            return img, target ,ind

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        elif self.mode == 'train' :
            return len(self.train_imgs)

if __name__ == '__main__':
    fooddata = Animal10N_dataset(root_dir='./Animal1N/', mode='train')