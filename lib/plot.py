from numpy import amax, amin, clip
from matplotlib import pyplot as plt


def plot(im, norm=False, doClip=True, save=False, fig=None):
    if norm is True:
        max = amax(im)
        min = amin(im)
        i = (im - min)/(max - min)
    elif doClip is True:
        i = clip(im, 0, 1)
    else:
        i = im
    if fig is None:
        plt.figure()
    plt.axis('off')
    if len(i.shape) == 3:
        plt.imshow(i)
    else:
        plt.imshow(i, cmap='gray')

    if save is not False:
        plt.imsave(save, i)
