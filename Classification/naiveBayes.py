# -*- coding: utf-8 -*-
"""ML in naive Bayes classifier
    contents:
    1.word list to vector func.
    2.naive Bayes classifier training func.
    3.naive Bayes classify func.
    4.Example: classifying spam email with naive Bayes
"""
from numpy import *


# 1.word list to vector func.
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea',
                    'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him',
                  'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute',
                  'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                  'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet |= set(document)
    return list(vocabSet)


def setOfwords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word : %s is not in my Vocabulary" % word
    return returnVec


# bag-of-words document model
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 2.naive Bayes classifier training func.
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    """
    p1Num = zeros(numWords)
    p0Num = zeros(numWords)
    modify by laplacian correction:
    """
    p1Num = ones(numWords)
    p0Num = ones(numWords)
    N = 2  # 训练集中可能类别数
    Ni = 2  # 第i个属性可能取值数，此例中全为2
    p0Denom = N
    p1Denom = Ni
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)  # log将乘变加
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


# 3.naive Bayes classify func.
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfwords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfwords2Vec(myVocabList, testEntry))
    print testEntry, 'classified:', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfwords2Vec(myVocabList, testEntry))
    print testEntry, 'classified:', classifyNB(thisDoc, p0V, p1V, pAb)


# 4.Example: classifying spam email with naive Bayes
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('C:/Users/dell/Desktop/pydata/mlaction/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('C:/Users/dell/Desktop/pydata/mlaction/Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfwords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfwords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print 'classification error: ', docList[docIndex]
    print 'the error rate is: ', float(errorCount)/len(testSet)


if __name__ == '__main__':
    spamTest()