# -*- coding: utf-8 -*-
"""Ml in action chapter7 Improving classification with the Adaboost
    contents:
    1.Decision stump-generating func.
    2.Adaboost training with decision stumps
    3.Example:Adaboost on a difficult dataset

    interpretation:
    用串行生成序列化的方法生成一系列基分类器,集成通过基学习器的线性组合实现
"""
from numpy import *


# 1.Decision stump-generating func.
def loadSimpData():
    dataMat = matrix([[1, 2.1], [2, 1.1],
                      [1.3, 1], [1, 1], [2, 1]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,  minError, bestClassEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D:", D.T
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst:", classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)  # classLabels与classEst不一致时给予更大值
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print "aggClassEst:", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print "total error:", errorRate, "\n"
        if errorRate == 0.0:
            break
    return weakClassArr


# 3.Example:Adaboost on a difficult dataset
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

if __name__ == '__main__':
    D = mat(ones((5, 1))/5)
    dataMat, classLabels = loadSimpData()
    # print adaBoostTrainDS(dataMat, classLabels, 9)
    datArr, labelArr = loadDataSet('C:\Users\dell\Desktop\pydata\mlaction\Ch07\horseColicTraining2.txt')
    print adaBoostTrainDS(datArr, labelArr, 10)