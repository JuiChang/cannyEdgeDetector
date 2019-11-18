import cv2
import numpy as np
import math
from spatialFiltering import gaussianFilter, sobelFilter, LoGFilter, adaptiveLocalNoiseReductionFilter
import matplotlib.pyplot as plt
from cannyRelated import nms, doubleThresholds, otsu
from intensityTransformation import normalizeFullDomain
import os


if __name__ == '__main__':
    inputFolder = "data"
    fileName = "p1im5.bmp"
    inputPath = os.path.join(inputFolder, fileName)
    fileNameSplit = os.path.splitext(fileName)[0]
    img = cv2.imread(inputPath)
    outputFolder = os.path.join("outputs", fileNameSplit)
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    cv2.imwrite(os.path.join(outputFolder, fileName), img)

    ################ preprocessing

    ################ edge detection
    size = 3
    mag, dir = sobelFilter(img, size)
    cv2.imwrite(os.path.join(outputFolder, "sobel_{}_mag.bmp".format(size)), mag)

    ################## Canny edge detection start from magnitude & direction maps
    nmsMag = nms(mag, dir)
    cv2.imwrite(os.path.join(outputFolder, "nmsMag.bmp"), nmsMag)

    # plt.hist(mag.ravel(), bins=100)
    # plt.show()

    thres = otsu(nmsMag)
    print("otsu thres:", thres)

    for i in range(1, 20, 2):
        for j in [9, 11]:
            iter = i
            seSize = j
            douThres = doubleThresholds(nmsMag, thres, thres / 2, False, iter, seSize, cross=True)
            douThresImg = douThres * 255
            cv2.imwrite(os.path.join(outputFolder, "sobel_douThres_{}_{}_cross.bmp".format(seSize, iter)), douThresImg)