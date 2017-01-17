from numpy import *
import operator

def createDataSet():
    group = [[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]]
    labels = ['A', 'A', 'B', 'B']
    return group, labels