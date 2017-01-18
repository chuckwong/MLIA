import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
from imp import *

def runRecognize(filename):
    reload(kNN)
    kNN.recognizeDigit(filename)

def runDigit():
    reload(kNN)
    kNN.handwritingClassTest()

def runPerson():
    reload(kNN)
    kNN.classifyPerson()

def testData():
    reload(kNN)
    kNN.datingClassTest()

def plotDatingData():
    reload(kNN)
    datingDataMat, datingLabels = kNN.file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(normMat[:, 0], normMat[:, 1], 15.0*array(datingLabels), 10.0*array(datingLabels))
    plt.xlabel("Flyier Miles Earned Per Year")
    plt.ylabel("Time Spend Playing Video Games")
    plt.title("Dating History")
    plt.legend()
    plt.show()