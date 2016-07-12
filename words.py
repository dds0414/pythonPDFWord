# -*- coding: utf-8 -*-

import Image
import cv2
from pylab import *
import os
import time


def guiyi(dataStr):
    rowIndex = []
    colIndex = []
    colLength = 0
    for index, x in enumerate(dataStr):
        rowTotal = 0
        colLength = len(x)
        for y in x:
            rowTotal += y
        if rowTotal > 0:
            rowIndex.append(index)
    for i in range(colLength):
        colTotal = 0
        for x in dataStr:
            colTotal += x[i]
        if colTotal > 0:
            colIndex.append(i)
    # res = np.array(data)[rowIndex[1]:rowIndex[-1], colIndex[1]:colIndex[-1]]
    return rowIndex, colIndex


def setImage(image):
    # 灰度处理
    grayMat = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 反转并去噪
    binMat = 255 - grayMat
    for indexRow, i in enumerate(binMat):
        for index, n in enumerate(i):
            if n < 25:
                binMat[indexRow][index] = 0
    # 去边缘
    row, col = guiyi(binMat)
    one = binMat[row[0]:row[-1], col[0]:col[-1] + 1]
    # resize
    res = cv2.resize(one, (30, 32), Image.ANTIALIAS)
    # 归一化
    data = res.reshape((1, 960))[0]
    return data, res


def imageTrain(filesPath):
    files = os.listdir(filesPath)
    for f in files:
        # 读取图像
        path = r"./image/" + f
        image = cv2.imread(path)
        data, res = setImage(image)
        # 存入train
        timeName = time.time()
        originFile = path.split('/')[2]
        fileName = originFile.split('.')[0]
        cv2.imwrite("./image2/" + fileName + "_" + str(timeName) + "1.jpg", res)
        np.savetxt("./train/" + fileName + ".txt", data)

if __name__ == '__main__':
    filePath = r"./image"
    imageTrain(filePath)

