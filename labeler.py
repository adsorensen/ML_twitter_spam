# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
"""
from featuresFile import *
from ID3 import *
import math

#size2 = 8

    

def main():
    file = './DatasetRetry/data-splits/data.train'
    fileEval = './DatasetRetry/data-splits/data.eval.anon'
    testFile = './DatasetRetry/data-splits/data.test'
    evalIds = './DatasetRetry/data-splits/data.eval.id'
    
    features = importFiles(file)
    fEval = importFiles(fileEval)
    test = importFiles(testFile) 
    
    
    
    maxDepth = 7
    
    r = ID3main(features, maxDepth)
    
    #testData(features, r, file)
    
    
    
    
    #testData(test, r, testFile)
    
    a, l = testData(fEval, r, fileEval)
    final = labelSubmission(l, evalIds)
    
    for f in final:
        print(f)
    
    #print(getDepth(r))
    #crossValidate()
    
    print('done')
    
def crossValidate():
    maxDepth = 6
    file00 = './Updated_Dataset/Updated_CVSplits/updated_training00.txt'
    file01 = './Updated_Dataset/Updated_CVSplits/updated_training01.txt'
    file02 = './Updated_Dataset/Updated_CVSplits/updated_training02.txt'
    file03 = './Updated_Dataset/Updated_CVSplits/updated_training03.txt'
    r00 = importFiles(file00)
    r01 = importFiles(file01)
    r02 = importFiles(file02)
    r03 = importFiles(file03)
    
    
    train01, s01 = mergeResults(r00, r01, r02)
    t01, ts01 = getData(r03)
    
    train02, s02 = mergeResults(r00, r01, r03)
    t02, ts02 = getData(r02)
    
    train03, s03 = mergeResults(r00, r02, r03)
    t03, ts03 = getData(r01)
    
    train04, s04 = mergeResults(r01, r02, r03)
    t04, ts04 = getData(r00)
    
    
    
    # test group 1
    r = ID3main(train01, maxDepth)
    print(getDepth(r))
    #testData(train01, r, s01, 'first training')
    ac1 = testData(t01, r, ts01, 'first test group')
    
    # test group 2
    r = ID3main(train02, maxDepth)
    print(getDepth(r))
    #testData(train02, r, s02, 'second training')
    ac2 = testData(t02, r, ts02, 'second group')
    
    #test group 3
    r = ID3main(train03, maxDepth)
    print(getDepth(r))
    #testData(train03, r, s03, 'third training')
    ac3 = testData(t03, r, ts03, 'third group')
    
    #test group 4
    r = ID3main(train04, maxDepth)
    print(getDepth(r))
    #testData(train04, r, s04, 'fourth training')
    ac4 = testData(t04, r, ts04, 'fourth group')
    
    acFinal = 0.0
    acFinal = (ac1 + ac2 + ac3 + ac4) / float(4)
    
    print(acFinal)
    
    
    
    
def mergeResults(r1, r2, r3):
    results = r1 + r2 + r3
    
    
    size = len(results)
    size2 = 8
    names = []
    labels = []
    features = [[0 for x in range(size)] for y in range(size2)]
    
    for r in results:
        
        str = r[0].strip().split(' ')
        name = " ".join(str[1:])
        labels.append(str[0])
        names.append(name)
        
    features = initializeFeatures(names, labels, features)
    return features, size
    
    
    
def traverseTree(data, r, size):
    n = r
    for i in range(0, size):
        e = n.key
        q = n.element
        if e == '0' or e =='1':
            if data[-1] == e:
                return 1
            else:
                return 0
        if data[e] == 1:
            n = n.yes
        else:
            n = n.no
            
def testData(features, r, file):
    correct = 0
    incorrect = 0
    labels = []
    size2 = len(features[0])
    size = len(features)
    for f in features:
        data = []
        data = f
        t = traverseTree(data, r, size2)
        if t == 1:
            labels.append(1)
            correct = correct + 1
        else:
            labels.append(0)
            incorrect = incorrect + 1
        
        del data[:]        
        
#==============================================================================
#     for i in range(0, size):
#         data = []
#         for f in features[0:]:
#             print(f)
#             data.append(f[i])
#             
#         
#         t = traverseTree(data, r, size2)
#         if t == 1:
#             correct = correct + 1
#         else:
#             incorrect = incorrect + 1
#         
#         del data[:]
#==============================================================================
        

    print('Stats for ' + file)
    print('correct= ' , correct, '   incorrect= ', incorrect)
    ac = 0.0
    ac = correct/float(size)
    print('accuracy: ', ac)
    print('')
    return ac, labels
    
    
    
def getDepth(root):
    d = height(root)
    return d
    
    
def height(n):
    if (n.yes == None and n.no == None):
        return 0
    else:
        return max(height(n.yes), height(n.no)) + 1
        
def getTestData(file):
    
    results = importFiles(file)
    size = len(results)
    size2 = 8
    names = []
    labels = []
    
    features = [[0 for x in range(size)] for y in range(size2)]
    
    for r in results:
        str = r[0].strip().split(' ')
        name = " ".join(str[1:])
        labels.append(str[0])
        names.append(name)
        
    features = initializeFeatures(names, labels, features)
    return features, size
    
main()