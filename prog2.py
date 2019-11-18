import cv2
import numpy as np
import math
from spatialFiltering import gaussianFilter, sobelFilter, LoGFilter
import matplotlib.pyplot as plt
from scipy import ndimage
import os


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


def doubleThresholds(mag, thres1, thres2, returnStrong=False, iter=1, seSize=3):
    """
    :param mag: 2D array storing gradient magnitude of each pixels
    :param thres1:
    :param thres2:
    :return: 2D bool array
    """
    strong = mag >= thres1
    if returnStrong:
        return strong
    weak = np.logical_and(mag < thres1, mag >= thres2)
    # cv2.imwrite("outputs/p1im4/sobel_weak.bmp", weak * 255)
    currStrong = strong
    thresWeak = weak
    for _ in range(iter):
        mask = ndimage.morphology.binary_dilation(currStrong, np.ones((seSize, seSize), dtype='bool'))
        thresWeak = np.logical_and(thresWeak, mask)
        currStrong = np.logical_or(currStrong, thresWeak)
    return currStrong


if __name__ == '__main__':
    inputFolder = "data/"
    fileName = "p1im6.bmp"
    inputPath = os.path.join(inputFolder, fileName)
    fileNameSplit = os.path.splitext(fileName)[0]
    img = cv2.imread(inputPath)
    outputFolder = os.path.join("outputs/", fileNameSplit)
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    cv2.imwrite(os.path.join(outputFolder, fileName), img)

    ################ preprocessing
    # img = gaussianFilter(img, 5)

    ################ edge detection

    # for size in [3, 5, 7]:
    for size in [3]:
        mag, dir = sobelFilter(img, size)
        # print(mag.min(), mag.max())
        cv2.imwrite(os.path.join(outputFolder, "sobel_{}_mag.bmp".format(size)), mag)
        # dirImg = (dir + math.pi) * 255 / (2 * math.pi)
        # cv2.imwrite(os.path.join(outputFolder, "sobel_{}_dir.bmp".format(size)), dirImg)

    ### Canny edge detection steps after sobel
    nmsMag = nms(mag, dir)
    cv2.imwrite(os.path.join(outputFolder, "nmsMag.bmp"), nmsMag)

    thres = otsu(nmsMag)
    print("otsu thres:", thres)
    cv2.imwrite(os.path.join(outputFolder, "my_thres.bmp"), (nmsMag >= thres) * 255)
    strong = doubleThresholds(nmsMag, thres * 2.3, thres, True)
    strongImg = strong * 255
    cv2.imwrite(os.path.join(outputFolder, "sobel_strong.bmp"), strongImg)
    # iter = 1
    # douThres = doubleThresholds(nmsMag, thres, thres / 2, False, iter, 21)
    # douThres = doubleThresholds(nmsMag, thres * 2, thres, False)
    # douThresImg = douThres * 255
    # cv2.imwrite(os.path.join(outputFolder, "sobel_douThres_{}.bmp".format(iter)), douThresImg)

    # plt.hist(mag.ravel(), bins=100)
    # plt.show()

    # for size in [3, 5, 7]:
    #     logImg = LoGFilter(img, size)
    #
    #     resultImg = 255 * (logImg > 0)
    #     cv2.imwrite(os.path.join(outputFolder, "LoG_{}_pos.bmp".format(size)), resultImg)
    #     # resultImg = 255 * (logImg < 0)
    #     # cv2.imwrite(os.path.join(outputFolder, "LoG_{}_neg.bmp".format(size)), resultImg)
    #     resultImg = 255 * (np.abs(logImg) < 3)
    #     cv2.imwrite(os.path.join(outputFolder, "LoG_{}_zero.bmp".format(size)), resultImg)
    #
    # for size in [3, 5, 7]:
    #     lapImg = LoGFilter(img, size, withoutGaussian=True)
    #
    #     resultImg = 255 * (lapImg > 0)
    #     cv2.imwrite(os.path.join(outputFolder, "Lap_{}_pos.bmp".format(size)), resultImg)
    #     # resultImg = 255 * (lapImg < 0)
    #     # cv2.imwrite(os.path.join(outputFolder, "Lap_{}_neg.bmp".format(size)), resultImg)
    #     resultImg = 255 * (np.abs(lapImg) < 3)
    #     cv2.imwrite(os.path.join(outputFolder, "Lap_{}_zero.bmp".format(size)), resultImg)

    # for size in [3, 5, 7]:
    #     resultImg = 255 * (LoGFilter(img, size, version=1) > 0)
    #     cv2.imwrite(os.path.join(outputFolder, "LoG_{}_v1.bmp".format(size)), resultImg)

