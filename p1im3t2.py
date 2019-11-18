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
    outputFolder = os.path.join("outputs", fileNameSplit + "t2")
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    cv2.imwrite(os.path.join(outputFolder, fileName), img)

    ################ preprocessing
    # img = gaussianFilter(img, 9)
    img = normalizeFullDomain(img, False)
    img = gammaCorrection(img, 0.5, True)
    img = adaptiveLocalNoiseReductionFilter(img, 11, 50, fileNameSplit)

    ################ edge detection

    for size in [3]:
        mag, dir = sobelFilter(img, size)
        cv2.imwrite(os.path.join(outputFolder, "sobel_{}_mag.bmp".format(size)), mag)

    ################## Canny edge detection start from magnitude & direction maps
    nmsMag = nms(mag, dir)
    cv2.imwrite(os.path.join(outputFolder, "nmsMag.bmp"), nmsMag)

    # plt.hist(mag.ravel(), bins=100)
    # plt.show()

    thres = otsu(nmsMag)
    print("otsu thres:", thres)

    # for dividend in range(1, 10):
    #     strong = doubleThresholds(nmsMag, thres / dividend, None, True)
    #     strongImg = strong * 255
    #     cv2.imwrite(os.path.join(outputFolder, "sobel_strong_{}.bmp".format(dividend)), strongImg)

    # for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
    # for i in [0.5]:
    # for i in range(1, 20, 2):
    #     for j in [9, 11]:
    iter = 1
    seSize = 7
    # douThres = doubleThresholds(nmsMag, thres * 1.3, thres * i, False, iter, seSize)
    douThres = doubleThresholds(nmsMag, thres / 2, thres / 3, False, iter, seSize)
    douThresImg = douThres * 255
    cv2.imwrite(os.path.join(outputFolder, "sobel_douThres_{}_{}.bmp".format(seSize, iter)), douThresImg)
