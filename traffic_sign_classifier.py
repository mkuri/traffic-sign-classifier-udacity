import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import explore_dataset
import vis_imgs

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


def main():
    train, valid, test, sign_names = load_files()
    explore_dataset.output_data_summery(train, valid, test)
    explore_dataset.output_histogram(train, valid, test)

    vis_imgs.output_grid_imgs('./figures/explore_imgs.png',
            5, 5, train['features'], train['labels'], sign_names)



if __name__ == "__main__":
    main()
