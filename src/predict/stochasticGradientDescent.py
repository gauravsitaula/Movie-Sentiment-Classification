from sklearn.linear_model import SGDClassifier as SGD
from sklearn.model_selection import GridSearchCV

import pickle
import pandas as pd

def pred():
    # Load fitted training data
    trainAfterFit = pickle.load(open("../picks/fittedTrainData.pkl","rb"))
    # Load prediction column
    predCol = pickle.load(open("../picks/predCol","rb"))
    # Load fitted test data
    testAfterFit = pickle.load(open("../picks/fittedTestData.pkl","rb"))
    # Load test data
    test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3 )
    
    # Constant that multiplies the regularization term. Defaults to 0.0001
    sgd_params = {'alpha': [0.00006, 0.00007, 0.00008, 0.0001, 0.0005]} 
    # Initialize SGD classifier
    modelSGD = GridSearchCV(
                             SGD(
                                 random_state = 0, # The seed of the pseudo random number generator to use when shuffling the data.
                                 shuffle = True, # Whether or not the training data should be shuffled after each epoch. Defaults to True.
                                 loss = 'modified_huber'
                                 
                                 # The loss function to be used. Defaults to 'hinge', which gives a linear SVM. 
                                 # The 'log' loss gives logistic regression, a probabilistic classifier. 
                                 # 'modified_huber' is another smooth loss that brings tolerance to outliers as well as probability estimates. 
                                 # 'squared_hinge' is like hinge but is quadratically penalized. 'perceptron' is the linear loss used by the perceptron algorithm. 
                                 # The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.
                                
                                 ), 
                             sgd_params,
                             scoring = 'roc_auc', # A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
                             cv = 20 # If an integer is passed, it is the number of folds.
                            ) 
    # Fit the classifier according to the given training data.
    modelSGD.fit(trainAfterFit,predCol) 
    
    print(modelSGD.cv_results_)
    '''
    Contains scores for all parameter combinations in param_grid. Each entry corresponds to one parameter setting. Each named tuple has the attributes:
    parameters, a dict of parameter settings
    mean_validation_score, the mean score over the cross-validation folds
    cv_validation_scores, the list of scores for each fold
    '''
    # Make prediction on fitted test data. These are Probability estimates. The returned estimates for all classes are ordered by the label of classes.
    SGDresult = modelSGD.predict_proba(testAfterFit)[:,1]
    # Create and store predictions in DataFrame and csv
    SGDoutput = pd.DataFrame(data={"id":test["id"], "sentiment":SGDresult})
    SGDoutput.to_csv('../results/SGDPredictions.csv', index = False, quoting = 3)

# if __name__ == '__main__':
#     main()