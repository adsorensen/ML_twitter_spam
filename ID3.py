# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
"""
import math

COUNT = 0

class Node:
    def __init__(self, key, value):
        self.key = key
        self.element = value
        self.yes = None
        self.no = None

    def getEl(self):
        return self.element

    def getYes():
        return self.yes
        
    def getNo():
        return self.no
        
def getAllPosNeg(features):
    size = len(features[1])
    p = 0
    n = 0
    for f in features:
        if f[-1] == '1':
            p = p + 1
        else:
            n = n + 1
            
    #neg = [[0 for x in range(n)] for y in range(size)]
    neg = [[0 for y in range(size)] for x in range(n)]
    #pos = [[0 for x in range(p)] for y in range(size)]
    pos = [[0 for y in range(size)] for x in range(p)]
    n = 0
    p = 0
    for i in range(0, len(features)):
        
        if features[i][-1] == '0':
            for t in range(0, size):
                neg[n][t] = features[i][t]
            n = n + 1
            
        else:
            for t in range(0, size):
                pos[p][t] = features[i][t]
            p = p + 1
    return pos, neg
    
def posNeg(attributes, A):
    size = len(attributes[0])
    n = 0
    p = 0
    for v in A:
        if v == 1:
            p = p + 1
        else:
            n = n + 1
            
    #neg = [[0 for x in range(n)] for y in range(size)]
    neg = [[0 for y in range(size)] for x in range(n)]

    #pos = [[0 for x in range(p)] for y in range(size)]
    pos = [[0 for y in range(size)] for x in range(p)]
    
    n = 0
    p = 0
    for i in range(0, len(attributes)):
        
        if A[i] == 1:
            for t in range(0, size):
                pos[p][t] = attributes[i][t]
            p = p + 1
            
        else:
            for t in range(0, size):
                neg[n][t] = attributes[i][t]
            n = n + 1
    size1 = len(pos)
    size2 = len(neg)
    return pos, neg
    
def getEntropy(pos, neg):
    psize = len(pos)
    nsize = len(neg)
    allsize = psize + nsize
    
    
    pfrac = (psize/float(allsize))
    nfrac = (nsize/float(allsize))
    if pfrac == 0 or nfrac == 0:
        return 0
    
    e = (pfrac * -1) * math.log(pfrac, 2) - nfrac * math.log(nfrac, 2)
    return e
    
def findCommonLabel(features):
    p = 0
    n = 0
    for f in features:
        if f[-1] == '1':
            p = p + 1
        else:
            n = n + 1
    if p >= n:
        return '1'
    else:
        return '0'
    

def ID3main(features, maxDepth):
    attributes = {0 : "length of screen name", 
                    1 : "length of description",
                    2 : "longevity: days",
                    3 : "longevity: hours",
                    4 : "longevity: minutes",
                    5 : "longevity: seconds",
                    6 : "number of following",
                    7 : "number of followers",
                    8 : "the ratio of following and followers",
                    9 : "the number of posted tweets",
                    10 : "the number of posted tweets per day",
                    11 : "the average number of links in tweets",
                    12 : "the average number of unique links in tweets",
                    13 : "the average numer of username in tweets",
                    14 : "the average numer of unique username in tweets",
                    15 : "the change rate of number of following"}
    global COUNT
    COUNT = 0
    
    root = ID3rec(features, attributes, maxDepth)
    return root
    
        
def ID3rec(features, attributes, maxDepth):
    global COUNT
    #print(COUNT)
    
    commonLabel = findCommonLabel(features)
    x = COUNT
    if x > maxDepth:
        return Node(commonLabel, None)
    
        
    pos, neg = getAllPosNeg(features)
    first = features[0][-1]
    sizeofData = len(features)
    flag = False
    for f in features:
        if f[-1] == first:
            flag = True
        else:
            flag = False
            break
    
    if(flag):
        COUNT = COUNT - 1
        return Node(first, None)
    else:
        # make root node
        if len(attributes) == 0:
            COUNT = COUNT - 1
            return Node(commonLabel, None)
        else:
            entropy = getEntropy(pos, neg)
            e = getBestAttribute(features, attributes, entropy)
            
            s = attributes.get(e)
            A = []
            for f in features:
                A.append(f[e])
            #A = features[e]
            
            newAttributes = dict()
            for k, v in attributes.items():
                if k == e:
                    pass
                else:
                    newAttributes[k] = v
            
            root = Node(e, s)
            
            # yes and no are sets divided whether A is 1 or 0
            yes, no = posNeg(features, A)
           
            if len(yes) == 0:
                COUNT = COUNT - 1
                root.yes = Node(commonLabel, None)
            else:
                COUNT = COUNT + 1
                root.yes = ID3rec(yes, newAttributes, maxDepth)
                
            if len(no) == 0:
                COUNT = COUNT - 1
                root.no = Node(commonLabel, None)
            else:
                COUNT = COUNT + 1
                root.no = ID3rec(no, newAttributes, maxDepth) 
                
            COUNT = COUNT - 1
            return root
        
 
def getBestAttribute(features, attributes, e):
    size = len(features[0])
    labels = []
    for f in features:
        labels.append(f[-1])
    
    infoGains = []
    testAttributes = list(attributes.keys())
    m = -0.2
    element = 1
    for i in range(1, size):
        if i in testAttributes:
            feature = []
            for f in features:
                feature.append(f[i])
            sf = []
            
            infoGains.append(getInfoGain(feature, labels, e))
            
            if m < max(infoGains):
                m = max(infoGains)
                element = i
        else:
            pass
    return element
        
    
def getInfoGain(feature, labels, eAll):
    size = len(feature)
    size2 = len(labels)
    

    # feature, label
    yy = 0
    yn = 0
    ny = 0
    nn = 0
    
    for i in range(0, size2):
        if (labels[i] == '1'):
            if feature[i] == 1:
                yy = yy + 1
            else:
                ny = ny + 1
        else:
            if feature[i] == 1:
                yn = yn + 1
            else:
                nn = nn + 1
    totalyes = yy + yn
    totalno = ny + nn
    
    if totalyes == 0:
        temp1 = 0
        temp2 = 0
    else:
        temp1 = yy/float(totalyes)
        temp2 = yn/float(totalyes)
    
    #print(yy, yn, ny, nn)
    
    
    if temp1 == 0 or temp2 == 0:
        e1 = 0
    else:
        e1 = (temp1 * -1) * math.log(temp1, 2) - (temp2 * math.log(temp2, 2))
    
    if totalno == 0:
        temp1 = 0
        temp2 = 0
    else:
        temp1 = ny/float(totalno)
        temp2 = nn/float(totalno)
    
    if temp1 == 0 or temp2 == 0:
        e2 = 0
    else:
        e2 = (temp1 * -1) * math.log(temp1, 2) - (temp2 * math.log(temp2, 2))
     
    temp1 = totalyes/float(size)
    temp2 = totalno/float(size)
    
    eExpected = (temp1 * e1) + (temp2*e2)
    infoGain = eAll - eExpected
    
    return infoGain
    
