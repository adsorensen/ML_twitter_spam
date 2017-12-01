# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
File containing all perceptron algorithms and helper functions.
Perceptrons: simple, dynamic, margin, average.
"""
from file_parser import *
import math

def simple_main(fv, e):
    b = .007
    w = initialize_weights(fv)
    # r = 1 or .1, .01
    r = 1
    w, b, c = s_adjust_v(fv, b, w, r, e)
    a = test_accuracy(fv, w, b)
    return a, w, b, c
    
def dynamic_main(fv, e):
    b = .0004
    w = initialize_weights(fv)
    r = .01
    w, b, c = d_adjust_v(fv, b, w, r, e)
    a = test_accuracy(fv, w, b)
    return a, w, b, c
    
    
def margin_main(fv, e, u):
    b = .0004
    w = initialize_weights(fv)
    r = .01
    w, b, c = m_adjust_v(fv, b, w, r, u, e)
    a, l = test_accuracy(fv, w, b)
    return a, w, b, c
    
def average_main(fv, e):
    b = .00008
    w = initialize_weights(fv)
    r = 1
    w, b, c = a_adjust_v(fv, b, w, r, e)
    a = test_accuracy(fv, w, b)
    return a, w, b, c
    
def a_adjust_v(fv, b, w, r, e):
    x = []
    a = w
    ba = 0
    c = 0
    for t in range(0, e):
        for f in fv:
            x = f[:-1]
            y = f[-1]
            dot = sum( [x[i]*w[i] for i in range(len(w))] ) + b
            
            if y*dot < 0:
                for i in range(0, len(w)):
                    w[i] = w[i] + r * (y * x[i])
                b = b + r*y
                c = c + 1
            a = [x + y for x, y in zip(a, w)]
            ba = ba + b
    return a, ba, c
    
    
    
def d_adjust_v(fv, b, w, r, e):
    x = []
    i = 0.0
    rn = r
    t = 0.0
    c = 0
    for p in range(0, e):
        
        for f in fv:
            x = f[:-1]
            y = f[-1]
            dot = sum( [x[i]*w[i] for i in range(len(w))] ) + b
            
            if y*dot < 0:
                for i in range(0, len(w)):
                    w[i] = w[i] + rn * (y * x[i])
                b = b + rn*y  
                c = c + 1
            rn = r / float(t + 1)
            t = t + 1
    return w, b, c
        
        
def s_adjust_v(fv, b, w, r, e):
    x = []
    c = 0
    for t in range(0, e):
        for f in fv:
            x = f[:-1]
            y = f[-1]
            dot = sum( [x[i]*w[i] for i in range(len(w))] ) + b
            
            if y*dot < 0:
                for i in range(0, len(w)):
                    w[i] = w[i] + r * (y * x[i])
                b = b + r*y
                c = c + 1
    return w, b, c
    
def m_adjust_v(fv, b, w, r, u, e):
    x = []
    rn = r
    t = 0.0
    c = 0
    for p in range(0, e):
        for f in fv:
            x = f[:-1]
            y = f[-1]
            dot = sum( [x[i]*w[i] for i in range(len(w))] ) + b
            
            if y*dot < u:
                for i in range(0, len(w)):
                    w[i] = w[i] + rn * (y * x[i])
                b = b + rn*y  
                c = c + 1
            rn = r / float(t + 1)
            t = t + 1
    return w, b, c
    
    
def test_accuracy(fv, w, b):
    x = []
    m = 0.0;
    labels = []
    total = float(len(fv))
    for f in fv:
        x = f[:-1]
        y = f[-1]
        dot = sum( [x[i]*w[i] for i in range(len(w))] ) + b
        
        if y*dot < 0:
            m = m + 1
            labels.append(0)
        else:
            labels.append(1)
            
    c = total - m
    return float(c) / total, labels
    
    
    
def cross_validate_simple():
    cross0 = 'Dataset/CVSplits/training00.data'
    cross1 = 'Dataset/CVSplits/training01.data'
    cross2 = 'Dataset/CVSplits/training02.data'
    cross3 = 'Dataset/CVSplits/training03.data'
    cross4 = 'Dataset/CVSplits/training04.data'
    
    r = 1
    e = 10
    
    
    r0 = importFile(cross0)
    r1 = importFile(cross1)
    r2 = importFile(cross2)
    r3 = importFile(cross3)
    r4 = importFile(cross4)
    
    train0 = initialize_vector(r0+r1+r2+r3)
    test0 = initialize_vector(r4)
    
    train1 = initialize_vector(r0+r1+r2+r4)
    test1 = initialize_vector(r3)
    
    train2 = initialize_vector(r0+r1+r3+r4)
    test2 = initialize_vector(r2)
    
    train3 = initialize_vector(r0+r2+r3+r4)
    test3 = initialize_vector(r1)
    
    train4 = initialize_vector(r1+r2+r3+r4)
    test4 = initialize_vector(r0)
    
    b0 = .00007
    b1 = .00007
    b2 = .00007
    b3 = .00007
    b4 = .00007
    
    w0 = initialize_weights(train0)
    w1 = initialize_weights(train1)
    w2 = initialize_weights(train2)
    w3 = initialize_weights(train3)    
    w4 = initialize_weights(train4)   
    
    w0, b0, c = s_adjust_v(train0, b0, w0, r, e)
    w1, b1, c = s_adjust_v(train1, b1, w1, r, e)
    w2, b2, c = s_adjust_v(train2, b2, w2, r, e)
    w3, b3, c = s_adjust_v(train3, b3, w3, r, e)
    w4, b4, c = s_adjust_v(train4, b4, w4, r, e)
    
    ac0 = test_accuracy(test0, w0, b0)
    ac1 = test_accuracy(test1, w1, b1)
    ac2 = test_accuracy(test2, w2, b2)
    ac3 = test_accuracy(test3, w3, b3)
    ac4 = test_accuracy(test4, w4, b4)
    
    final = (ac0 + ac1 + ac2 + ac3 + ac4) / float(5)
    return final
    
    
def cross_validate_dynamic():
    cross0 = 'Dataset/CVSplits/training00.data'
    cross1 = 'Dataset/CVSplits/training01.data'
    cross2 = 'Dataset/CVSplits/training02.data'
    cross3 = 'Dataset/CVSplits/training03.data'
    cross4 = 'Dataset/CVSplits/training04.data'
    
    r = .01
    e = 10
    
    r0 = importFile(cross0)
    r1 = importFile(cross1)
    r2 = importFile(cross2)
    r3 = importFile(cross3)
    r4 = importFile(cross4)
    
    train0 = initialize_vector(r0+r1+r2+r3)
    test0 = initialize_vector(r4)
    
    train1 = initialize_vector(r0+r1+r2+r4)
    test1 = initialize_vector(r3)
    
    train2 = initialize_vector(r0+r1+r3+r4)
    test2 = initialize_vector(r2)
    
    train3 = initialize_vector(r0+r2+r3+r4)
    test3 = initialize_vector(r1)
    
    train4 = initialize_vector(r1+r2+r3+r4)
    test4 = initialize_vector(r0)
    
    b0 = .00007
    b1 = .00007
    b2 = .00007
    b3 = .00007
    b4 = .00007
    
    w0 = initialize_weights(train0)
    w1 = initialize_weights(train1)
    w2 = initialize_weights(train2)
    w3 = initialize_weights(train3)    
    w4 = initialize_weights(train4)   
    
    w0, b0, c = d_adjust_v(train0, b0, w0, r, e)
    w1, b1, c = d_adjust_v(train1, b1, w1, r, e)
    w2, b2, c = d_adjust_v(train2, b2, w2, r, e)
    w3, b3, c = d_adjust_v(train3, b3, w3, r, e)
    w4, b4, c = d_adjust_v(train4, b4, w4, r, e)
    
    ac0 = test_accuracy(test0, w0, b0)
    ac1 = test_accuracy(test1, w1, b1)
    ac2 = test_accuracy(test2, w2, b2)
    ac3 = test_accuracy(test3, w3, b3)
    ac4 = test_accuracy(test4, w4, b4)
    
    final = (ac0 + ac1 + ac2 + ac3 + ac4) / float(5)
    
    return final
    
    
def cross_validate_margin(u, r):
    cross0 = 'Dataset/CVSplits/training00.data'
    cross1 = 'Dataset/CVSplits/training01.data'
    cross2 = 'Dataset/CVSplits/training02.data'
    cross3 = 'Dataset/CVSplits/training03.data'
    cross4 = 'Dataset/CVSplits/training04.data'
    
    e = 10
    
    r0 = importFile(cross0)
    r1 = importFile(cross1)
    r2 = importFile(cross2)
    r3 = importFile(cross3)
    r4 = importFile(cross4)
    
    train0 = initialize_vector(r0+r1+r2+r3)
    test0 = initialize_vector(r4)
    
    train1 = initialize_vector(r0+r1+r2+r4)
    test1 = initialize_vector(r3)
    
    train2 = initialize_vector(r0+r1+r3+r4)
    test2 = initialize_vector(r2)
    
    train3 = initialize_vector(r0+r2+r3+r4)
    test3 = initialize_vector(r1)
    
    train4 = initialize_vector(r1+r2+r3+r4)
    test4 = initialize_vector(r0)
    
    b0 = .00007
    b1 = .00007
    b2 = .00007
    b3 = .00007
    b4 = .00007
    
    w0 = initialize_weights(train0)
    w1 = initialize_weights(train1)
    w2 = initialize_weights(train2)
    w3 = initialize_weights(train3)    
    w4 = initialize_weights(train4)   
    
    w0, b0, c = m_adjust_v(train0, b0, w0, r, u, e)
    w1, b1, c = m_adjust_v(train1, b1, w1, r, u, e)
    w2, b2, c = m_adjust_v(train2, b2, w2, r, u, e)
    w3, b3, c = m_adjust_v(train3, b3, w3, r, u, e)
    w4, b4, c = m_adjust_v(train4, b4, w4, r, u, e)
    
    ac0 = test_accuracy(test0, w0, b0)
    ac1 = test_accuracy(test1, w1, b1)
    ac2 = test_accuracy(test2, w2, b2)
    ac3 = test_accuracy(test3, w3, b3)
    ac4 = test_accuracy(test4, w4, b4)
    
    final = (ac0 + ac1 + ac2 + ac3 + ac4) / float(5)
    
    return final
    
    
def cross_validate_average(r):
    cross0 = 'Dataset/CVSplits/training00.data'
    cross1 = 'Dataset/CVSplits/training01.data'
    cross2 = 'Dataset/CVSplits/training02.data'
    cross3 = 'Dataset/CVSplits/training03.data'
    cross4 = 'Dataset/CVSplits/training04.data'
    
    
    e = 10
    
    r0 = importFile(cross0)
    r1 = importFile(cross1)
    r2 = importFile(cross2)
    r3 = importFile(cross3)
    r4 = importFile(cross4)
    
    train0 = initialize_vector(r0+r1+r2+r3)
    test0 = initialize_vector(r4)
    
    train1 = initialize_vector(r0+r1+r2+r4)
    test1 = initialize_vector(r3)
    
    train2 = initialize_vector(r0+r1+r3+r4)
    test2 = initialize_vector(r2)
    
    train3 = initialize_vector(r0+r2+r3+r4)
    test3 = initialize_vector(r1)
    
    train4 = initialize_vector(r1+r2+r3+r4)
    test4 = initialize_vector(r0)
    
    b0 = .00007
    b1 = .00007
    b2 = .00007
    b3 = .00007
    b4 = .00007
    
    w0 = initialize_weights(train0)
    w1 = initialize_weights(train1)
    w2 = initialize_weights(train2)
    w3 = initialize_weights(train3)    
    w4 = initialize_weights(train4)   
    
    w0, b0, c = a_adjust_v(train0, b0, w0, r, e)
    w1, b1, c = a_adjust_v(train1, b1, w1, r, e)
    w2, b2, c = a_adjust_v(train2, b2, w2, r, e)
    w3, b3, c = a_adjust_v(train3, b3, w3, r, e)
    w4, b4, c = a_adjust_v(train4, b4, w4, r, e)
    
    ac0 = test_accuracy(test0, w0, b0)
    ac1 = test_accuracy(test1, w1, b1)
    ac2 = test_accuracy(test2, w2, b2)
    ac3 = test_accuracy(test3, w3, b3)
    ac4 = test_accuracy(test4, w4, b4)
    
    final = (ac0 + ac1 + ac2 + ac3 + ac4) / float(5)
    
    return final


    
# function initializes weight vector to 0's
def initialize_weights(fv):
    w = []
    r = 0
    for i in range(0, len(fv[0])-1):
        w.append(r)
    return w
    
def majority(fv):
    p = 0
    n = 0  
    total = len(fv)
    for f in fv:
        if f[-1] == 1:
            p = p + 1
        else:
            n = n + 1
    ma = max(n, p)
    f = float(ma)/float(total)
    return f
    
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