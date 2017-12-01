# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
"""
from ID3 import *

def initializeFeatures(names, labels, features):
    i = 0
    # initialize names to first slot
    for d in features[0]:
        features[0][i] = names[i]
        i = i+1
    
    # get labels in last spot
    i = 0    
    for f in features[7]:
        features[7][i] = labels[i]
        i = i + 1
    
    features[1] = firstNameLonger(features[0], features[1])
    features[2] = middleName(names, features[2])
    features[3] = firstLastLetter(names, features[3])
    features[4] = firstBeforeLast(names, features[4])
    features[5] = secondLetterVowel(names, features[5])
    features[6] = evenLast(names, features[6])
    
    return features
    
def getData(results):
    size = len(results)
    size2 = 17
    names = []
    labels = []
    max1 = 0.0
    min1 = 500.0
    total = 0.0
    #features = [[0 for x in range(size)] for y in range(size2)]
    features = [[0 for y in range(size2)] for x in range(size)]
    i = 0
    for r in results:
        features[i][16] = r[0]
        features[i][0] = int(r[1][2:])
        features[i][1] = int(r[2][2:])
        features[i][2] = int(r[3][2:])
        features[i][3] = int(r[4][2:])
        features[i][4] = int(r[5][2:])
        features[i][5] = int(r[6][2:])
        features[i][6] = int(r[7][2:])
        features[i][7] = int(r[8][2:])
        features[i][8] = float(r[9][2:])
        features[i][9] = int(r[10][3:])
        features[i][10] = float(r[11][3:])
        features[i][11] = float(r[12][3:])
        features[i][12] = float(r[13][3:])
        features[i][13] = float(r[14][3:])
        features[i][14] = float(r[15][3:])
        features[i][15] = float(r[16][3:])
        i = i + 1
        #for rr in r:
        
    i = 0
    f = 7
    for r in features:
        if r[f] > max1:
            max1 = r[f]
        if r[f] < min1:
            min1 = r[f]
        total = total + r[f]
        i = i + 1
    avg = float(total / i)
#==============================================================================
#     print(avg)
#     print(max1)
#     print(min1)
#==============================================================================
    # uncomment out for binary features    
    features = binaryFeature(features)
    #findInfoGains(features)
    
    return features
    
    
    
def findInfoGains(features):
    labels = getLabels(features)
    pos, neg = getAllPosNeg(features)
    e = getEntropy(pos, neg)
    ft = 15
    big = 0
    split = 0
    
        
    for i in range(-5, 2000, 1):
        feature = []
        ss = 0.0
        ss = float(i)/1000
        for f in features:
            feature.append(f[ft])
        feature = splitFeature(feature, ss)
        
        g = getInfoGain(feature, labels, e)
        if (g > big):
            big = g
            split = ss
            
    print("Feature: ", ft, " info gain: ", big, " split: ", split)
    
    
    
    
def splitFeature(feature, split):
    size = len(feature)
    for i in range(0, size):
        if feature[i] > split:
            feature[i] = 1
        else:
            feature[i] = 0
    
    return feature
    
def binaryFeature(features):
    #splits
    s = [9, 75, 42, 19, 52, 33, 237, 293, 
         1.0, 190, 2, .33, .326, .069, .0197, .306]
         
    #print(features[0])
    for r in features:
#==============================================================================
#         if r[-1] == '1':
#             r[-1] = '+'
#         else:
#             r[-1] = '-'
#==============================================================================
#==============================================================================
#         if r[1] > s[0]:
#             r[1] = 1
#         else:
#             r[1] = 0
#==============================================================================
        for i in range(0, 16):
            if r[i] > s[i]:
                r[i] = 1
            else:
                r[i] = 0
            
    #print(features[0])
    return features
    
    

def importFiles(file):
    results = []
    with open(file) as inputFile:
        for l in inputFile:
            results.append(l.strip().split(' '))
            
    features = getData(results)
    return features
    
    
def getLabels(features):
    labels = []
    for f in features:
        labels.append(f[-1])
        
    return labels


def labelSubmission(l, file):
    results = []
    final = []
    size = len(l)
    
    with open(file) as inputFile:
        for a in inputFile:
            results.append(a.strip().split(' '))
            
    for i in range(0, size):
        final.append(str(results[i][0]) + ',' + str(l[i]))
            
    return final
