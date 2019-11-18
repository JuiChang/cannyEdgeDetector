import cv2
import numpy as np
import math
from spatialFiltering import gaussianFilter, sobelFilter, LoGFilter, adaptiveLocalNoiseReductionFilter
import matplotlib.pyplot as plt
from cannyRelated import nms, doubleThresholds, otsu
from intensityTransformation import normalizeFullDomain, gammaCorrection
import os


if __name__ == '__main__':
    inputFolder = "data"
    fileName = "p1im3.bmp"
    inputPath = os.path.join(inputFolder, fileName)
    fileNameSplit = os.path.splitext(fileName)[0]
    img = cv2.imread(inputPath)
    outputFolder = os.path.join("outputs", fileNameSplit)
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    cv2.imwrite(os.path.join(outputFolder, fileName), img)

    ################ preprocessing
    # img = gaussianFilter(img, 9)
    img = normalizeFullDomain(img, False)
    img = gammaCorrection(img, 0.5, True)
    img = adaptiveLocalNoiseReductionFilter(img, 11, 50, fileNameSplit)

    ################ edge detection

    # for size in [3, 5, 7]:
    for size in [3]:
        mag, dir = sobelFilter(img, size)
        cv2.imwrite(os.path.join(outputFolder, "sobel_{}_mag.bmp".format(size)), mag)
        # dirImg = (dir + math.pi) * 255 / (2 * math.pi)
        # cv2.imwrite(os.path.join(outputFolder, "sobel_{}_dir.bmp".format(size)), dirImg)

    thres = otsu(mag)
    print("otsu thres:", thres)
    # for thresSet in [1.5, 1.3, 1, 0.9, 0.7, 0.5, 0.3, 0.1]:
    for thresSet in [0.3]:
        thresMag = 255 * (mag > thres * thresSet)
        cv2.imwrite(os.path.join(outputFolder, "sobel_thresMag_{}.bmp".format(thresSet)), thresMag)

    # for size in [3]:
    #     logImg = LoGFilter(img, size)
    #
    #     resultImg = 255 * np.logical_and(logImg > 3, logImg < 5)
    #     cv2.imwrite(os.path.join(outputFolder, "LoG_{}_pos_.bmp".format(size)), resultImg)
    #     # resultImg = 255 * (logImg < 0)
    #     # cv2.imwrite(os.path.join(outputFolder, "LoG_{}_neg.bmp".format(size)), resultImg)
    #     resultImg = 255 * (np.abs(logImg) < 3)
    #     cv2.imwrite(os.path.join(outputFolder, "LoG_{}_zero.bmp".format(size)), resultImg)
    #
    # for size in [3]:
    #     lapImg = LoGFilter(img, size, withoutGaussian=True)
    #
    #     resultImg = 255 * np.logical_and(lapImg > 5, lapImg < 10)
    #     cv2.imwrite(os.path.join(outputFolder, "Lap_{}_pos_.bmp".format(size)), resultImg)
    #     # resultImg = 255 * (lapImg < 0)
    #     # cv2.imwrite(os.path.join(outputFolder, "Lap_{}_neg.bmp".format(size)), resultImg)
    #     resultImg = 255 * (np.abs(lapImg) < 3)
    #     cv2.imwrite(os.path.join(outputFolder, "Lap_{}_zero.bmp".format(size)), resultImg)

    # ################## Canny edge detection start from magnitude & direction maps
    # nmsMag = nms(mag, dir)
    # cv2.imwrite(os.path.join(outputFolder, "nmsMag.bmp"), nmsMag)
    #
    # # plt.hist(mag.ravel(), bins=100)
    # # plt.show()
    #
    # thres = otsu(nmsMag)
    # print("otsu thres:", thres)
    #
    # # for dividend in range(1, 10):
    # #     strong = doubleThresholds(nmsMag, thres / dividend, None, True)
    # #     strongImg = strong * 255
    # #     cv2.imwrite(os.path.join(outputFolder, "sobel_strong_{}.bmp".format(dividend)), strongImg)
    #
    # # for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
    # # for i in [0.5]:
    # # for i in range(1, 20, 2):
    # #     for j in [9, 11]:
    # iter = 1
    # seSize = 3
    # # douThres = doubleThresholds(nmsMag, thres * 1.3, thres * i, False, iter, seSize)
    # douThres = doubleThresholds(nmsMag, thres, thres / 2, False, iter, seSize, cross=True)
    # douThresImg = douThres * 255
    # cv2.imwrite(os.path.join(outputFolder, "sobel_douThres_{}_{}_cross.bmp".format(seSize, iter)), douThresImg)
