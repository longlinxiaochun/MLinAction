# -*- coding: utf-8 -*-
"""ML in action chapter 8 linear regression P154
    contents:
 1.linear regression
 2.locally weighted linear regression
 3.ridge regression
 4.forward stagewise regression
 5.Example:forecasting the price of LEGO sets
"""

# 1.linear regression
# use equation to get a linear regression
from numpy import *


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))-1
    dataMat = []
    labelMel = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine =line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMel.append(float(curLine[-1]))
    return dataMat, labelMel


def standardRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular , cannot do inverse "
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

# use gradient descent to get a linear regression


def costfunction(theta, xarr, yarr):
    m = len(yarr)
    xMat = c_[ones(m, 1), mat.xarr]
    yMat = mat(yarr)
    J = (theta * xMat.T - yMat) ** 2
    cost = 1 / (2 * m) * J.sum()
    return cost


def gradientdescent(alpha, theta, xarr, yarr, num_iter):
    m = len(yarr)
    xMat = mat(xarr)
    yMat = mat(yarr)
    Jhistory = []
    thetas = theta
    for i in range(1, num_iter):
        temp = thetas * xMat.T - yMat
        theta[0] -= alpha / m * temp.sum()
        theta[1] -= alpha / m * (temp * xMat[:, 1]).sum()
        thetas = theta
        # theta -= alpha * temp.sum() is not right because theta need update at the same time
        Jhistory.append(costfunction(theta, xarr, yarr))
    return theta, Jhistory


# 5.Example:forecasting the price of LEGO sets

"""  shopping information retrieval function"""
from time import sleep
import json
import urllib2


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    """prevent making too many API calls too quickly
    but still rise an error <urlopen error [Errno 10054] >"""

    myAPIstr = 'get from code.google.com'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?\
                  key=%s&county=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condittion'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" %\
                          (yr, numPce, newFlag, origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print 'problem with item %d' % i


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5196, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    filename = "C:\Users\dell\Desktop\pydata\mlaction\Ch08\ex0.txt"
    xArr, yArr = loadDataSet(filename)
    ws = standardRegres(xArr, yArr)
    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
