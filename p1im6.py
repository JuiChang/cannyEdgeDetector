import cv2
import numpy as np
import math
from spatialFiltering import gaussianFilter, sobelFilter, LoGFilter, adaptiveLocalNoiseReductionFilter
import matplotlib.pyplot as plt
from cannyRelated import nms, doubleThresholds, otsu
import os


if __name__ == '__main__':
    inputFolder = "data"
    fileName = "p1im6.bmp"
    inputPath = os.path.join(inputFolder, fileName)
    fileNameSplit = os.path.splitext(fileName)[0]
    img = cv2.imread(inputPath)
    outputFolder = os.path.join("outputs/", fileNameSplit)
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    cv2.imwrite(os.path.join(outputFolder, fileName), img)

    ################ preprocessing
    # img = gaussianFilter(img, 9)
    img = adaptiveLocalNoiseReductionFilter(img, 9, 100, fileNameSplit)

    ################ edge detection

    # for size in [3, 5, 7]:
    # for size in [3, 7]:
    size = 7
    mag, dir = sobelFilter(img, size)
    # print(mag.min(), mag.max())
    cv2.imwrite(os.path.join(outputFolder, "sobel_{}_mag_ada7.bmp".format(size)), mag)
    # dirImg = (dir + math.pi) * 255 / (2 * math.pi)
    # cv2.imwrite(os.path.join(outputFolder, "sobel_{}_dir.bmp".format(size)), dirImg)

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

    ################## Canny edge detection start from magnitude & direction maps
    nmsMag = nms(mag, dir)
    cv2.imwrite(os.path.join(outputFolder, "nmsMag_ada7.bmp"), nmsMag)

    # plt.hist(mag.ravel(), bins=100)
    # plt.show()

    thres = otsu(nmsMag)
    print("otsu thres:", thres)

    strong = doubleThresholds(nmsMag, thres, thres, True)
    strongImg = strong * 255
    cv2.imwrite(os.path.join(outputFolder, "sobel_strong_ada7.bmp"), strongImg)

    # for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for i in [0.5]:
        iter = 1
        seSize = 5
        # douThres = doubleThresholds(nmsMag, thres * 1.3, thres * i, False, iter, seSize)
        douThres = doubleThresholds(nmsMag, thres, thres * i, False, iter, seSize)
        douThresImg = douThres * 255
        cv2.imwrite(os.path.join(outputFolder, "sobel_douThres_{}_{}_{}_ada7.bmp".format(iter, seSize, i)), douThresImg)

