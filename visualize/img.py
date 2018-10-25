from pathlib import Path
from typing import Iterable, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from matplotlib import rcParams
rcParams['font.family'] = 'Liberation Sans'
rcParams['font.size'] = 6

def combine_in_one_img(imgs: Iterable[np.ndarray],
                       titles: Iterable[str],
                       cmaps: Iterable[Optional[str]],
                       layout: str) -> plt.Figure:
    fig = plt.figure()
    
    for i, (img, title, cmap) in enumerate(zip(imgs, titles, cmaps)):
        ax = fig.add_subplot(int(layout+str(i+1)))
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)

    return fig


def combine_funced_imgs(img: np.ndarray,
                        funcs: Iterable[Callable[[np.ndarray], np.ndarray]],
                        titles: Iterable[str],
                        cmaps: Iterable[Optional[str]],
                        layout: str) -> plt.Figure:
    fig = plt.figure()

    for i, (func, title, cmap) in enumerate(zip(funcs, titles, cmaps)):
        ax = fig.add_subplot(int(layout+str(i+1)))
        ax.imshow(func(img), cmap=cmap)
        ax.set_title(title)

    return fig
