# import datetime
# from numpy import *
# import math
# # from numarray import *
# #import pylab as pl
#
# #
# def loadData(fileName):
#     dataMat = []
#     labelMat = []
#     temp_temp = []
#     temp = []
#     i = 0
#     with open('C:/Users/Administrator/Desktop/ex0.txt') as txtFile:
#         for line in txtFile.readlines():
#              # labelMat.append(float, line.split())[-1])
#             temp = line.split('\n')
#             temp = temp[0].split('    ')
#             # dataMat.append([])
#             dataMat.append([float(temp[1])])
#             labelMat.append([float(temp[2])])
#             # dataMat[i].append(float(temp[2]))
#             # dataMat.append(float(temp[1]),float(temp[2]))
#             # dataMat.append(map(float, line.split('    '))[0:-1])
#     return dataMat, labelMat
#
#
# #
# def standRegres(dataSet, labelSet):
#     xMat = mat(dataSet)
#     yMat = mat(labelSet).T
#     xTx = xMat.T * xMat
#     if linalg.det(xTx) is 0.0:
#         print ("Warning !!!")
#         return
#     ws = xTx.I * (xMat.T * yMat)
#     return ws #
#
# #
# def lwlr(testPoint, xArr, yArr, k):
#     m = shape(xArr)
#     weight = eye((m))
#     for i in range(m):
#         error = xArr[i, :] - testPoint
#         weight[i, i] = math.exp((error * error.T) / (-2.0 * k ** 2)) #
#     xTWx = xArr.T * (weight * xArr)
#     if linalg.det(xTWx) is 0.0:
#         print ("Warning !!!")
#         return
#     ws = xTWx.I * (xArr.T * (weight * yArr))
#     return testPoint * ws #
#
# #
# def lwlrTest(xArr, yArr, k=1):
#     yPre = zeros(shape(xArr)[0])
#     # print "k is:", k
#     len = shape(xArr)[0]
#     for i in range(len):
#         yPre[i] = lwlr(xArr[i], xArr, yArr, k)
#     return yPre # yPre
#
# #
# # def outPic(xArr, yArr, yPre, theta, k):
# #     #
# #     theta = theta.tolist()
# #     pl.xlim(-0.1, 1.1, 0.1)
# #     pl.ylim(2.5, 5, 0.1)
# #     pl.scatter(xArr[:, -1], yArr, s=8, color='red', alpha=1)
# #     x = arange(-0.1, 1.1, 0.1)
# #     yHat = theta[0] + theta[1] * x  #
# #     pl.plot(x, yHat, '-')
# #     pl.show()
# #     #
# #     pl.scatter(xArr[:, 1], yArr, s=8, color='red', alpha=1)
# #     xArr = mat(xArr)
# #     srtInd = xArr[:, 1].argsort(0)
# #     xSort = xArr[srtInd][:, 0, :]
# #     pl.plot(xSort[:, 1], yPre[srtInd], '-')
# #     pl.xlabel("k = %.2f" % k)
# #     pl.show()
#
#
# if __name__ == '__main__':
#     #
#     past = datetime.datetime.now()
#     xArr, yArr = loadData("ex0.txt")
#     theta = standRegres(xArr, yArr)
#     xArr = mat(xArr)
#     yArr = mat(yArr).T
#     k = 0.03  #
#     yPre = lwlrTest(xArr, yArr, k)  #
#     # outPic(xArr, yArr, yPre, theta, k) #
#     #
#     print ("time:"), datetime.datetime.now() - past

# coding:utf-8

import datetime
from numpy import *

#import pylab as pl

#
def loadData(fileName):
    dataMat = []
    labelMat = []
    with open('C:/Users/Administrator/Desktop/data.txt') as txtFile:
        for line in txtFile.readlines():
            labelMat.append(map(float, line.split())[-1])
            dataMat.append(map(float, line.split())[0:-1])
    return dataMat, labelMat


#
def standRegres(dataSet, labelSet):
    xMat = mat(dataSet)
    yMat = mat(labelSet).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) is 0.0:
        print "Warning !!!"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws #

#
def lwlr(testPoint, xArr, yArr, k):
    m = shape(xArr)[0]
    weight = eye((m))
    for i in range(m):
        error = xArr[i, :] - testPoint
        weight[i, i] = math.exp((error * error.T) / (-2.0 * k ** 2)) #
    xTWx = xArr.T * (weight * xArr)
    if linalg.det(xTWx) is 0.0:
        print "Warning !!!"
        return
    ws = xTWx.I * (xArr.T * (weight * yArr))
    return testPoint * ws #

#
def lwlrTest(xArr, yArr, k=1):
    yPre = zeros(shape(xArr)[0])
    # print "k is:", k
    len = shape(xArr)[0]
    for i in range(len):
        yPre[i] = lwlr(xArr[i], xArr, yArr, k)
    return yPre # yPre

# #
# def outPic(xArr, yArr, yPre, theta, k):
#     #
#     theta = theta.tolist()
#     pl.xlim(-0.1, 1.1, 0.1)
#     pl.ylim(2.5, 5, 0.1)
#     pl.scatter(xArr[:, -1], yArr, s=8, color='red', alpha=1)
#     x = arange(-0.1, 1.1, 0.1)
#     yHat = theta[0] + theta[1] * x  #
#     pl.plot(x, yHat, '-')
#     pl.show()
#
#     pl.scatter(xArr[:, 1], yArr, s=8, color='red', alpha=1)
#     xArr = mat(xArr)
#     srtInd = xArr[:, 1].argsort(0)
#     xSort = xArr[srtInd][:, 0, :]
#     pl.plot(xSort[:, 1], yPre[srtInd], '-')
#     pl.xlabel("k = %.2f" % k)
#     pl.show()


if __name__ == '__main__':
    past = datetime.datetime.now()
    xArr, yArr = loadData("data.txt")
    theta = standRegres(xArr, yArr)
    xArr = mat(xArr)
    yArr = mat(yArr).T
    k = 0.03
    yPre = lwlrTest(xArr, yArr, k)
    # outPic(xArr, yArr, yPre, theta, k)

    print "time:", datetime.datetime.now() - past