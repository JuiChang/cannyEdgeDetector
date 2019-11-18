import numpy as np


def histogramEqual(img, lowerb=0, upperb=255):
    grayImg = 0.3 * img[:, :, 2] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 0]
    grayImg = grayImg.astype("uint8")

    grayHistogram = np.zeros(256)

    ### histogram of the gray-scale image
    for i in range(255):
        grayHistogram[i] = (grayImg == i).sum()

    ### normalize the area to 1
    grayHistogram = grayHistogram / grayHistogram.sum()

    ### mapping to [0, 255]
    # grayMappingFunc = np.cumsum(grayHistogram) * 255
    grayMappingFunc = np.cumsum(grayHistogram) * (upperb - lowerb) + lowerb

    grayResult = grayMappingFunc[grayImg]
    mappingMultiplier = grayResult / grayImg
    return img * np.repeat(mappingMultiplier[:, :, np.newaxis], 3, axis=2)


def gammaCorrection(img, gamma, respectively=True):
    """ the input image must has 3 channels """

    if respectively:
        imgFloat = img.astype("float64")
        for i in range(0, imgFloat.shape[2]):
            imgFloat[:, :, i] = imgFloat[:, :, i] / 255

        c = 1
        correct = lambda t: c * t ** gamma
        vfunc = np.vectorize(correct)
        result = vfunc(imgFloat)

        for i in range(0, imgFloat.shape[2]):
            result[:, :, i] = result[:, :, i] * 255

        return result

    grayImg = 0.3 * img[:, :, 2] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 0]
    grayImg = np.clip(grayImg, a_min=None, a_max=255)
    normalizedGrayImg = grayImg / 255

    c = 1
    correct = lambda t: c * t ** gamma
    vfunc = np.vectorize(correct)
    grayImgResult = vfunc(normalizedGrayImg)
    grayImgResult = grayImgResult * 255
    ratio = grayImgResult / (grayImg + 0.00001)

    return np.clip(img * np.repeat(ratio[:, :, np.newaxis], 3, axis=2), a_min=None, a_max=255)


def normalizeFullDomain(img, respectively=False):
    """ the input image must has 3 channels """

    if respectively:
        result = np.zeros(img.shape)
        for i in range(img.shape[2]):
            result[:, :, i] = (img[:, :, i] - img[:, :, i].min()).astype('int') * 255 / (
                        img[:, :, i].max() - img[:, :, i].min() + 0.00001)
            # print((img[:, :, i] - img[:, :, i].min()) * 255)
            # print((img[:, :, i] - img[:, :, i].min()) * 256)
        return result

    grayImg = 0.3 * img[:, :, 2] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 0]
    grayImg = np.clip(grayImg, a_min=None, a_max=255)
    noralizedGrayImg = (grayImg - grayImg.min()) * 255 / (grayImg.max() - grayImg.min() + 0.00001)
    ratio = noralizedGrayImg / grayImg
    return np.clip(img * np.repeat(ratio[:, :, np.newaxis], 3, axis=2), a_min=None, a_max=255)