# -*- coding: utf-8 -*-
"""Ml in action chapter 5 logistic regression P83
    contents:
    1.logregression gradient ascent optimization functions and decision boundary plot
    2.stochastic gradient ascent and its modified way
    3.Example :estimating horse fatalities from colic
"""

# 1.logregression gradient ascent optimization functions and decision boundary plot
from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open(r'C:\Users\dell\Desktop\pydata\mlaction\Ch05\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = labelMat - h
        weights += alpha * dataMatrix.transpose() * error
    return weights  # (n,1)


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    wei = mat(wei)
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 4.0, 1.0)  # x=[-3,..,3]
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 2.stochastic stochastic gradient ascent and its modified way


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(dataMatrix[i]*weights.T)
        error = classLabels[i] - h
        weights += alpha * error * dataMatrix[i]
    return weights.reshape(n, 1)  # (1,n)


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    dataIndex = range(m)
    for j in range(numIter):
        for i in range(m):
            alpha = 4/(1+j+i)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(dataMatrix[randIndex]*weights.T)
            error = classLabels[randIndex] - h
            weights += alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
            return weights.reshape(n, 1)  # (1,n)


# Example :estimating horse fatalities from colic


def classifyVector(inX, weights):
    inX = mat(inX)
    prob = inX * weights
    if prob > 0:
        return 1
    else:
        return 0


def colicTest():
    frTrain = open("C:\Users\dell\Desktop\pydata\mlaction\Ch05\horseColicTraining.txt")
    frTest = open("C:\Users\dell\Desktop\pydata\mlaction\Ch05\horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    # trainWeights = gradAscent(array(trainingSet), trainingLabels)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is :" \
          "%f" % (numTests, errorSum/float(numTests))

if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # ws = gradAscent(dataArr, labelMat)
    ws = stocGradAscent1(dataArr, labelMat)
    plotBestFit(ws)
    multiTest()  # overflow encountered in exp
