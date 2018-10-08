import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Liberation Sans'
rcParams['font.size'] = 6


def output_data_summery(train, valid, test):
    print('>>> Output data summery ...')
    n_train = train['labels'].shape[0]
    n_valid = valid['labels'].shape[0]
    n_test = test['labels'].shape[0]

    image_shape = train['features'].shape[1:]
    n_classes = len(np.unique(train['labels']))

    text = 'Number of training examples = ' + str(n_train) + '\n'
    text += 'Number of validating examples = ' + str(n_valid) + '\n'
    text += 'Number of testing examples = ' + str(n_test) + '\n'
    text += 'Image data shape = ' + str(image_shape) + '\n'
    text += 'Number of classes = ' + str(n_classes) + '\n'

    with open('./outputs/data_summery.txt', 'w') as f:
        f.write(text)


def output_histogram(train, valid, test):
    print('>>> Output histogram ...')
    y_train = train['labels']
    y_valid = valid['labels']
    y_test = test['labels']

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    fig.subplots_adjust(hspace=0.5)

    def _set_data(y, ax, title):
        name, n = np.unique(y, return_counts=True)
        ax.bar(name, n)
        ax.set_title(title)

    _set_data(y_train, ax1, 'Train dataset histogram')
    _set_data(y_valid, ax2, 'Valid dataset histogram')
    _set_data(y_test, ax3, 'Test dataset histogram')

    fig.savefig('./figures/histogram.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
