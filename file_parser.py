# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
Helper functions to read in data, build lists, and initialize vectors.
"""


def importFile(file):
    results = []
    with open(file) as inputFile:
        for l in inputFile:
            results.append(l.strip().split(' '))
            
    features = get_data(results)
    return features
    
def get_data(results):
    size = len(results)
    size2 = len(results[0])
    i = 0
    features = [[0 for x in range(size2)] for y in range(size)]
    
    for r in results:
        if r[0] == '1':
            features[i][16] = 1
        else:
            features[i][16] = -1
        #features[i][16] = r[0]
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
    
    return features