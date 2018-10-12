import os, sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import cv2

from visualize.img import combine_in_one_img

def correct_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    img = np.uint8(255.0 * (img / 255.0)**gamma)
    return img


def normalize_luminance_srgb(img: np.ndarray,
                             cliplimit=2.0,
                             grid=(4, 4),
                             f_fig=None) -> np.ndarray:
    gamma = 2.2

    # sRGB -> Linear RGB
    lrgb = correct_gamma(img, 1.0/gamma)
    
    ycrcb = cv2.cvtColor(lrgb, cv2.COLOR_RGB2YCrCb)

    ycrcb_g = correct_gamma(ycrcb, gamma)

    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=grid)
    ycrcb_g[:,:,0] = clahe.apply(ycrcb_g[:,:,0])

    ycrcb_norm = correct_gamma(ycrcb_g, 1.0/gamma)

    rgb_norm = cv2.cvtColor(ycrcb_norm, cv2.COLOR_YCrCb2RGB)

    rgb_norm_g = correct_gamma(rgb_norm, gamma)

    if f_fig == True:
        imgs = [img, lrgb, ycrcb, ycrcb_norm, rgb_norm, rgb_norm_g]
        titles = ['Original(sRGB)', 'Linear RGB', 'YCrCb', 'Normalized YCrCb', 'Normalized Linear RGB', 'Normalized sRGB']
        cmaps = [None, None, None, None, None, None]
        layout = '23'
        fig = combine_in_one_img(imgs, titles, cmaps, layout)

        return (rgb_norm_g, fig)

    else:
        return rgb_norm_g
