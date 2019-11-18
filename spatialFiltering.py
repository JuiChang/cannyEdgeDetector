import numpy as np
import math
import pickle

filters2d = {
    "sobel_vertical_3": np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]),
    "sobel_vertical_5": np.array([[-2/8, -1/5,  0,  1/5,  2/8],
                        [-2/5, -1/2,  0,  1/2,  2/5],
                        [-2/4, -1/1,  0,  1/1,  2/4],
                        [-2/5, -1/2,  0,  1/2,  2/5],
                        [-2/8, -1/5,  0,  1/5,  2/8]]),
    "sobel_vertical_7": np.array([[-3/18, -2/13, -1/10, 0,  1/10, 2/13, 3/18],
                        [-3/13, -2/8,  -1/5,  0,  1/5,  2/8,  3/13],
                        [-3/10, -2/5,  -1/2,  0,  1/2,  2/5,  3/10],
                        [-3/9,  -2/4,  -1/1,  0,  1/1,  2/4,  3/9],
                        [-3/10, -2/5,  -1/2,  0,  1/2,  2/5,  3/10],
                        [-3/13, -2/8,  -1/5,  0,  1/5,  2/8,  3/13],
                        [-3/18, -2/13, -1/10, 0,  1/10, 2/13, 3/18]]),
    "laplacian_3": np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]]),
    "laplacian_5": np.array([[-4, -1, 0, -1, -4],
                             [-1, 2, 3, 2, -1],
                             [0, 3, 4, 3, 0],
                             [-1, 2, 3, 2, -1],
                             [-4, -1, 0, -1, -4]]),
    "laplacian_7": np.array([[-10, -5, -2, -1, -2, -5, -10],
                             [-5, 0, 3, 4, 3, 0, -5],
                             [-2, 3, 6, 7, 6, 3, -2],
                             [-1, 4, 7, 8, 7, 4, -1],
                             [-2, 3, 6, 7, 6, 3, -2],
                             [-5, 0, 3, 4, 3, 0, -5],
                             [-10, -5, -2, -1, -2, -5, -10]])
}


def correlation(img, filter, repeat=True):
    """
    valid input depths:
    img: 1, filter: 1
    img: 3, filter: 3
    img: 3, filter: 1
    """

    ### filter must be a square with odd H and W
    if filter.shape[0] != filter.shape[1] or filter.shape[0] % 2 == 0 or filter.shape[0] % 2 == 0:
        print("correlation: invalid filter!!")
        return
    reshape = False
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        reshape = True
    if len(filter.shape) == 2:
        filter = filter[:, :, np.newaxis]
    if img.shape[2] == 3 and filter.shape[2] == 1:
        filter = np.repeat(filter, 3, axis=2)

    pad = int((filter.shape[0] - 1) / 2)
    size = filter.shape[0]

    result = np.zeros(img.shape)
    paddedImg = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, img.shape[2]))
    paddedImg[pad: -pad, pad: -pad, :] = img
    if repeat:
        paddedImg[:pad, pad: -pad, :] = img[0, :, :].reshape((1, img.shape[1], img.shape[2]))
        paddedImg[-pad:, pad: -pad, :] = img[-1, :, :].reshape((1, img.shape[1], img.shape[2]))
        paddedImg[pad: -pad, :pad, :] = img[:, 0, :].reshape((img.shape[0], 1, img.shape[2]))
        paddedImg[pad: -pad, -pad:, :] = img[:, -1, :].reshape((img.shape[0], 1, img.shape[2]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j, :] = np.sum(paddedImg[i: i + size, j: j + size, :] * filter, axis=(0, 1))

    if reshape:
        img = np.reshape(img, (img.shape[0], img.shape[1]))
        result = np.reshape(result, (result.shape[0], result.shape[1]))

    return result


def calculateGaussianFilter(size, version, sd=5):
    """
    :param sd: the standard deviation, is only used in version 0
    """

    if version == 0:
        centerIndex = size // 2

        xIndex = np.tile(np.arange(size), (size, 1))
        xDiff = xIndex - centerIndex  # x diff with the center
        yIndex = np.tile(np.arange(size).reshape(-1, 1), (1, size))
        yDiff = yIndex - centerIndex  # y diff with the center
        numerator = xDiff ** 2 + yDiff ** 2

        filter2d = (1 / (2 * math.pi * sd ** 2)) * np.exp(-(numerator / (2 * sd ** 2)))

        filter2d = filter2d / np.sum(filter2d)

        return filter2d

    N = (size - 1) / 2

    filter1d = np.arange(-N, N + 1)
    numerator = (3 * filter1d / N) ** 2
    filter1d = np.exp(-(numerator / 2))
    filter1d = filter1d / np.sum(filter1d)

    filter2d = np.tile(filter1d, (size, 1)) * np.tile(filter1d.reshape(-1, 1), (1, size))

    return filter2d


def gaussianFilter(img, size, version=0, sd=5):
    """
    :param img: could be gray scale or 3-channel image
    """
    filter = calculateGaussianFilter(size, version, sd)
    return correlation(img, filter)


def sobelFilter(img, size):
    """
    :param img: could be gray scale or 3-channel BGR image
    :param size: must be either 3, 5, or 7
    """
    print("sobel filter")

    global filters2d

    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img = 0.11 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.3 * img[:, :, 2]

    filterV = filters2d["sobel_vertical_" + str(size)]
    filterH = filterV.T
    xGrad = correlation(img, filterV)
    yGrad = correlation(img, filterH)
    mag = np.abs(xGrad) + np.abs(yGrad)
    dir = np.arctan2(yGrad, xGrad)
    return mag, dir


def LoGFilter(img, size, version=0, sd=5, withoutGaussian=False):
    """
    :param img: could be gray scale or 3-channel BGR image
    :param size: must be either 3, 5, or 7
    """
    print("LoG filter")

    global filters2d

    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img = 0.11 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.3 * img[:, :, 2]

    filter = filters2d["laplacian_" + str(size)]
    if not withoutGaussian:
        filter = filter * calculateGaussianFilter(size, version, sd)
    return correlation(img, filter)


# def prewittFilter(img):


def adaptiveLocalNoiseReductionFilter(img, size, estNoise, imgName="imgName"):
    """ imgName is for .pkl saving """

    pklName = "alnrf_" + str(imgName) + "_" + str(size) + ".pkl"
    try:
        diffFromAvg = pickle.load(open(pklName, "rb"))
    except (OSError, IOError) as e:

        tmpShape = list(img.shape)
        tmpShape.append(size * size)
        tmpShape = tuple(tmpShape)
        diffFromAvg = np.zeros(tmpShape)

        for i in range(size * size):
            print("iter: ({}/{})".format(str(i), str(size * size)))
            subFilter = np.zeros(size * size)
            subFilter[i] = 1
            subFilter = np.reshape(subFilter, (size, size))
            filter = subFilter - np.ones((size, size)) / (size * size)
            diffFromAvg[..., i] = correlation(img, filter)

        pickle.dump(diffFromAvg, open(pklName, "wb"))

    localVarImg = np.sum(diffFromAvg ** 2, axis=-1) / (size * size)

    # print(localVarImg[:10, :10, 0])

    localMean = correlation(img, np.ones((size, size)) / (size * size))

    factor = estNoise / (localVarImg + 0.00001)
    factor = np.clip(factor, a_min=None, a_max=1)

    return img - factor * (img - localMean)


def bilateralFilter(img, size, sdg, imgName="imgName"):
    """
    valid input depths:
    img: 1, filter: 1
    img: 3, filter: 3
    img: 3, filter: 1
    imgName is for .pkl saving
    """

    filter = calculateGaussianFilter(size, 0)
    reshape = False
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        reshape = True
    if len(filter.shape) == 2:
        filter = filter[:, :, np.newaxis]
    if img.shape[2] == 3 and filter.shape[2] == 1:
        filter = np.repeat(filter, 3, axis=2)

    print("var:")
    print(np.var(img[:, :, 0]))
    print(np.var(img[:, :, 1]))
    print(np.var(img[:, :, 2]))

    pklName = "bilateral_" + str(imgName) + "_" + str(size) + ".pkl"
    try:
        weightMul = pickle.load(open(pklName, "rb"))
    except (OSError, IOError) as e:

        ### calculate the weight multiplier for each pixel
        weightMul = np.zeros((img.shape[0], img.shape[1], size, size, img.shape[2]))
        subFilterSubtractor = np.zeros((size, size))
        subFilterSubtractor[size // 2, size // 2] = 1
        for i in range(size):
            for j in range(size):
                print("iter: ({}/{})".format(str(i * size + j), str(size * size)))
                subFilter = np.zeros((size, size))
                subFilter[i, j] = 1
                tmpFilter = subFilter - subFilterSubtractor
                weightMul[:, :, i, j, :] = correlation(img, tmpFilter)

        pickle.dump(weightMul, open(pklName, "wb"))

    weightMul = np.exp(-1 * weightMul ** 2 / (2 * sdg))

    pad = int((filter.shape[0] - 1) / 2)
    size = filter.shape[0]

    result = np.zeros(img.shape)
    paddedImg = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, img.shape[2]))
    paddedImg[pad: -pad, pad: -pad, :] = img

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            weight = filter * weightMul[i, j, ...]
            weightDivisor = np.sum(weight, axis=(0, 1))
            weightDivisor = weightDivisor[np.newaxis, np.newaxis, :]
            weightDivisor = np.repeat(weightDivisor, size, axis=0)
            weightDivisor = np.repeat(weightDivisor, size, axis=1)
            result[i, j, :] = np.sum(paddedImg[i: i + size, j: j + size, :] * weight / weightDivisor, axis=(0, 1))

            # if i < 1 and j < 1:
            #     print((weight / weightDivisor)[:2, :2, 0])

    if reshape:
        img = np.reshape(img, (img.shape[0], img.shape[1]))
        result = np.reshape(result, (result.shape[0], result.shape[1]))

    return result
