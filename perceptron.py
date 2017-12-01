# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
Main driver for ASG 2: Perceptrons
"""
from file_parser import *
from perceptron_algo import *

def main():
    file = './DatasetRetry/data-splits/data.train'
    file2 = './DatasetRetry/data-splits/data.eval.anon'
    test = './DatasetRetry/data-splits/data.test'
    evalIDS = './DatasetRetry/data-splits/data.eval.id'

    
    fvTrain = importFile(file)
    fvEval = importFile(file2)
    fvTest = importFile(test)
    
    
    

    
    
#==============================================================================
#     # Prints out accuracies for different perceptron algorithms. 
#     a, w, b, c = simple_main(fvTrain, 15)
#     print("Simple perceptron with r = 1, epochs = 15")
#     print('Simple percpetron train accuracy ', a)
#     aa = test_accuracy(fvTrain, w, b)
#     print('Simple percpetron test accuracy ', aa)
#     print('Simple percpetron updates made: ', c)
#     print("\n")
#     
#     a, w, b, c = dynamic_main(fv, 17)
#     print("Dynamic perceptron with r = .01, epochs = 17")
#     print('Dynamic percpetron train accuracy ', a)
#     aa = test_accuracy(fvt, w, b)
#     print('Dynamic percpetron test accuracy ', aa)
#     print('Dynamic percpetron updates made: ', c)
#     print("\n")
#==============================================================================
    
    a, w, b, c = margin_main(fvTrain, 3, .01)
    print("Margin perceptron with r = .01, epochs = 14, u = .01")
    print('Margin percpetron train accuracy ', a)
    aa, l = test_accuracy(fvEval, w, b)
    print('Margin percpetron test accuracy ', aa)
    print('Margin percpetron updates made: ', c)
    print("\n")  
    final = labelSubmission(l, evalIDS)
    
    for f in final:
        print(f)
    
    
#==============================================================================
#     a, w, b, c = average_main(fv, 12)
#     print("Average perceptron with n = 1, epochs = 12")
#     print('Average percpetron train accuracy ', a)
#     aa = test_accuracy(fvt, w, b)
#     print('Average percpetron test accuracy ', aa)
#     print('Average percpetron updates made: ', c)
#     print("\n")
# 
#     f = majority(fvd)
#     ff = majority(fvt)
#     print('Majority baseline accuracy on dev: ', f)
#     print('Majority baseline accuracy on test: ', f)  
#==============================================================================
    
    
    
    
main()