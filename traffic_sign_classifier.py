import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import explore_dataset
from preprocess.img.normalize import normalize_luminance_srgb
from output_sign_imgs import output_grid_imgs, output_compared_imgs
from models.lenet import Lenet
from dataset import ImageDataset


def load_files():
    print('>>> Loading files ...')

    with open('./data/train.p', 'rb') as f:
        train = pickle.load(f)
    with open('./data/valid.p', 'rb') as f:
        valid = pickle.load(f)
    with open('./data/test.p', 'rb') as f:
        test = pickle.load(f)
    sign_names = pd.read_csv('./data/signnames.csv')

    return train, valid, test, sign_names


def preprocess(imgs):
    print('>>> Preprocessing ...')
    preprocessed = []

    def _grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def _normalize(img):
        return img.astype(np.float32)/255.0

    def _reshape(img):
        img = img.reshape(32, 32, 1)
        return np.transpose(img, (2, 0, 1))

    def _pipeline(img):
        img = normalize_luminance_srgb(img)
        img = _grayscale(img)
        img = _normalize(img)
        img = _reshape(img)
        return img

    for img in imgs:
        pipelined = _pipeline(img)
        preprocessed.append(pipelined)

    preprocessed = np.array(preprocessed)
    print(type(preprocessed))
    print(preprocessed.shape)

    return preprocessed


def train_model(model, train):
    print('>>> Train model ...')
    dataset = ImageDataset(train['features'], train['labels'])
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=2)
    n_epoch = 1
    lr = 0.01

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epoch):
        model.train()
        running_loss = 0.0

        for i, (features, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                        (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0


def main():
    train, valid, test, sign_names = load_files()
    # explore_dataset.output_data_summery(train, valid, test)
    # explore_dataset.output_histogram(train, valid, test)

    # vis_imgs.output_grid_imgs('./figures/explore_imgs.png',
    #         5, 5, train['features'], train['labels'], sign_names)

    # train['features'] = preprocess(train['features'])
    # with open('./data/train_preprocessed.p', 'wb') as f:
    #     pickle.dump(train, f)

    with open('./data/train_preprocessed.p', 'rb') as f:
        train = pickle.load(f)

    model = Lenet()
    train_model(model, train)


if __name__ == "__main__":
    main()
