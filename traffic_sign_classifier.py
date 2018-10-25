import pickle
import time
from pathlib import Path

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
from visualize.img import combine_in_one_img


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
    n_epoch = 10
    lr = 0.0005

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

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                        (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0

    return model


def test_model(model, test):
    print('>>> Test model ...')
    model.eval()
    dataset = ImageDataset(test['features'], test['labels'])
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=False, num_workers=2)
    
    correct = 0
    total = 0
    for i, (features, labels) in enumerate(dataloader):
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy: %.3f' % accuracy)

    return accuracy


def test_web_imgs(model, features):
    print('>>> Test web images ...')
    dataset = ImageDataset(features, np.array([13, 22, 15, 4, 38]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    model.eval()

    for i, (features, labels) in enumerate(dataloader):
        outputs = model(features)
        print('label', labels)
        print('outputs', outputs)
        _, predicted = torch.max(outputs, 1)
        m = torch.nn.Softmax()
        softmax = m(outputs)
        print('softmax', softmax)
        print('predicted', predicted)
        sorts = []
        for i, output in enumerate(softmax[0]):
            sorts.append((output.item(), i))
        sorts = sorted(sorts, key=lambda x: x[0], reverse=True)
        print(sorts)
        for i in range(5):
            print('%.4f' % sorts[i][0])

    
        
    


def load_web_imgs(paths):
    print('>>> Loading web images ...')
    
    imgs = []
    for path in paths:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    imgs = np.array(imgs)
    print(type(imgs))
    print(imgs.shape)

    fig = combine_in_one_img(imgs, ["", "", "", "", ""], [None, None, None, None, None], '15')
    fig.savefig('./figures/web_images.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    
    return imgs


def main():
    train, valid, test, sign_names = load_files()
    # explore_dataset.output_data_summery(train, valid, test)
    # explore_dataset.output_histogram(train, valid, test)

    # train['features'] = preprocess(train['features'])
    # output_grid_imgs('./outputs/grayscaled.png',
    #         5, 5, train['features'], train['labels'], sign_names)
    # with open('./data/train_preprocessed.p', 'wb') as f:
    #     pickle.dump(train, f)

    # with open('./data/train_preprocessed.p', 'rb') as f:
    #     train = pickle.load(f)
    
    model = Lenet()
    # model = train_model(model, train)
    #
    # torch.save(model.state_dict(), './outputs/lenet.p')

    model.load_state_dict(torch.load('./outputs/lenet.p'))

    # test['features'] = preprocess(test['features'])
    # valid['features'] = preprocess(valid['features'])

    imgs = load_web_imgs(Path('./data/web').glob('*'))
    features = preprocess(imgs)

    test_web_imgs(model, features)


    
    # test_model(model, test)



if __name__ == "__main__":
    main()
