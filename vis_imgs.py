import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Liberation Sans'
rcParams['font.size'] = 6

def display_grid(nrows, ncols, images, labels, sign_names, cmap=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    
    for row in range(nrows):
        for col in range(ncols):
            index = np.random.randint(len(images))
            image = images[index]
            label = labels[index]
            name = sign_names.query('ClassId=={}'.format(label))['SignName'].values[0]
            axes[row, col].imshow(image, cmap=cmap)
            axes[row, col].axis('off')
            axes[row, col].set_title(name)

    return fig


def output_grid_imgs(filename, nrows, ncols,
                     images, labels, sign_names, cmap=None):
    fig = display_grid(nrows, ncols, images, labels, sign_names, cmap)
    fig.savefig(filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
