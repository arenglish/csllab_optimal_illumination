from numpy import sqrt, sum, square, amin, amax


def deltaE(lab1, lab2, axis=2):
    return sqrt(sum(square(lab1 - lab2), axis=axis))


def deltaRGB(rgb1, rgb2, axis=2):
    return sqrt(sum(square(rgb1 - rgb2), axis=axis))


RGB_TO_GRAYSCALE_RATIOS = [0.2125, 0.7153, 0.0721]


def rgb2gray(rgb):
    g = rgb*RGB_TO_GRAYSCALE_RATIOS
    return g[:, :, 0] + g[:, :, 1] + g[:, :, 2]


def michelson_contrast(gray1, gray2):
    return (max([gray1, gray2]) - min([gray1, gray2]))/(max([gray1, gray2]) + min([gray1, gray2]))


def clip_rgb(rgb):
    _rgb = rgb.copy()
    _rgb[_rgb > 1] = 1
    _rgb[_rgb < 0] = 0
    return _rgb


def norm(rgb):
    max = amax(rgb)
    min = amin(rgb)

    return (rgb - min)/(max - min)
