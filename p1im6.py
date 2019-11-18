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
    img = adaptiveLocalNoiseReductionFilter(img, 9, 100, fileNameSplit)

    ################ edge detection
    size = 7
    mag, dir = sobelFilter(img, size)
    cv2.imwrite(os.path.join(outputFolder, "sobel_{}_mag_ada7.bmp".format(size)), mag)

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

    for i in [0.5]:
        iter = 1
        seSize = 5
        douThres = doubleThresholds(nmsMag, thres, thres * i, False, iter, seSize)
        douThresImg = douThres * 255
        cv2.imwrite(os.path.join(outputFolder, "sobel_douThres_{}_{}_{}_ada7.bmp".format(iter, seSize, i)), douThresImg)

