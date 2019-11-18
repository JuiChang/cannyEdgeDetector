import cv2
import numpy as np
import math
from scipy import ndimage

def nms(mag, dir):
    """
    :param mag: 2D array
    :param dir: [-pi, pi], 2D map
    :return: 2D array
    """

    ### round the grad direction
    # to 0~360 deg
    dir = (dir + math.pi) * 360 / (2 * math.pi)
    dir8 = dir // 45 + 1 * (dir % 45 > 22.5)
    dir8 = dir8 - 1 * (dir8 == 8)

    angleDiff = np.zeros((mag.shape[0], mag.shape[1], 8))
    paddedMag = np.zeros((mag.shape[0] + 2, mag.shape[1] + 2))
    paddedMag[1: -1, 1: -1] = mag
    angleDiff[:, :, 0] = mag - paddedMag[1: -1, 2:]
    angleDiff[:, :, 1] = mag - paddedMag[: -2, 2:]
    angleDiff[:, :, 2] = mag - paddedMag[: -2, 1: -1]
    angleDiff[:, :, 3] = mag - paddedMag[: -2, : -2]
    angleDiff[:, :, 4] = mag - paddedMag[1: -1, : -2]
    angleDiff[:, :, 5] = mag - paddedMag[2:, : -2]
    angleDiff[:, :, 6] = mag - paddedMag[2:, 1: -1]
    angleDiff[:, :, 7] = mag - paddedMag[2:, 2:]

    # if each position of dir8 corresponds to 0, 1, 2, or 3 (the 3rd index), should we remain or suppress
    remain = np.zeros((mag.shape[0], mag.shape[1], 4))
    for i in range(4):
        remain[:, :, i] = np.logical_and(angleDiff[:, :, i] > 0, angleDiff[:, :, i+4] > 0)
    lookRemain = dir8 % 4
    mask = np.zeros((mag.shape[0], mag.shape[1]), dtype='bool')
    for i in range(4):
        mask = np.logical_or(mask, np.logical_and(remain[:, :, i], lookRemain == i))

    return mag * mask


def otsu(map):
    """
    :param map: 2D np array, all element >= 0
    :return: the threshold
    """
    map = map.astype('int')

    # hist = np.histogram(map, bins=map.max()+1, range=(0, map.max()))
    numBin = map.max() + 1
    hist = np.zeros(numBin)
    for i in range(numBin):
        hist[i] = np.sum(map == i)
    hist = hist / map.size

    histCumuSum = np.zeros(numBin)
    histCumuSum[0] = hist[0]
    for i in range(1, numBin):
        histCumuSum[i] = histCumuSum[i - 1] + hist[i]

    histCumuMean = np.zeros(numBin)
    histCumuMean[0] = 0
    for i in range(1, numBin):
        histCumuMean[i] = histCumuMean[i - 1] + hist[i] * i

    globalMean = histCumuMean[-1]

    varBtw = np.zeros(numBin)
    for i in range(1, numBin):
        varBtw[i] = (globalMean * histCumuSum[i] - histCumuMean[i])**2 / ((histCumuSum[i]) * (1 - histCumuSum[i]))

    return np.argmax(varBtw)


def doubleThresholds(mag, thres1, thres2, returnStrong=False, iter=1, seSize=3, cross=False):
    """
    :param mag: 2D array storing gradient magnitude of each pixels
    :param thres1: larger threshold
    :param thres2: smaller threshold
    :return: 2D bool array
    """
    strong = mag >= thres1
    if returnStrong:
        return strong
    weak = np.logical_and(mag < thres1, mag >= thres2)
    cv2.imwrite("outputs/p1im6/sobel_weak.bmp", weak * 255)

    selected = strong
    for _ in range(iter):
        if cross:
            mask = ndimage.morphology.binary_dilation(selected, np.ones((seSize, 1), dtype='bool'))
            mask = np.logical_or(mask, ndimage.morphology.binary_dilation(selected, np.ones((1, seSize), dtype='bool')))
        else:
            mask = ndimage.morphology.binary_dilation(selected, np.ones((seSize, seSize), dtype='bool'))
        selectedWeak = np.logical_and(weak, mask)
        selected = np.logical_or(selected, selectedWeak)
    return selected