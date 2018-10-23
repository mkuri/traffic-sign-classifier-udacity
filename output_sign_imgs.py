import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'Liberation Sans'
rcParams['font.size'] = 6


def output_grid_imgs(filename, nrows, ncols, images, labels,
                     sign_names, cmap=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
    
    for row in range(nrows):
        for col in range(ncols):
            index = np.random.randint(len(images))
            image = images[index]
            label = labels[index]
            name = sign_names.query('ClassId=={}'.format(label))['SignName'].values[0]
            axes[row, col].imshow(image, cmap=cmap)
            axes[row, col].axis('off')
            axes[row, col].set_title(name)

    fig.savefig(fname=filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)


def output_compared_imgs(filename, nrows, befores, afters,
                         labels, sign_names, cmap=None):
    fig, axes = plt.subplots(nrows, 2, figsize=(4, 2*nrows))

    for row in range(nrows):
        index = np.random.randint(len(befores))
        before = befores[index]
        after = afters[index]
        label = labels[index]
        name = sign_names.query('ClassId=={}'.format(label))['SignName'].values[0]
        axes[row, 0].imshow(before, cmap=cmap)
        axes[row, 0].axis('off')
        axes[row, 0].set_title(name)
        axes[row, 1].imshow(after, cmap=cmap)
        axes[row, 1].axis('off')
        axes[row, 1].set_title(name)

    fig.savefig(fname=filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
