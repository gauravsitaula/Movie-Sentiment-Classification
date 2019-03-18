'''
Created on Jan 14, 2016

@author: abtpst
'''

from util import vectorize as vz
from predict import logisticRegression as lr, multinomialNaiveBayes as mnb, stochasticGradientDescent as sgd

if __name__ == '__main__':
    
    # First, transform training and test data
    #vz.vect()
    #Now classify
    #mnb.pred()

    lr.pred('This is the worst movie')
    #sgd.pred()