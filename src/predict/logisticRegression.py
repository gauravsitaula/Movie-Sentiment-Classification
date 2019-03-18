from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
import pickle
import pandas as pd
from util import vectorize as vz

# Load fitted training data
trainAfterFit = pickle.load(open("../picks/fittedTrainData.pkl","rb"))
# Load prediction column
predCol = pickle.load(open("../picks/predCol","rb"))
# Load fitted test data
testAfterFit = pickle.load(open("../picks/fittedTestData.pkl","rb"))
# Load test data
test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3 )

grid_values = {'C':[30]} # Decide which settings you want for the grid search.

# C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

#GridSearchCV implements a Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.fit" method and a Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.predict" method like any classifier except that the parameters of the classifier used to predict is optimized by cross-validation.

modelLR = GridSearchCV(
                        LR
                            (
                                penalty = 'l2', # Used to specify the norm used in the penalization. http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/

                                # One of the prime differences between Lasso and ridge regression is that in ridge regression, as the penalty is increased, all parameters are reduced while
                                # still remaining non-zero, while in Lasso, increasing the penalty will cause more and more of the parameters to be driven to zero. This is an advantage of
                                # Lasso over ridge regression, as driving parameters to zero deselects the features from the regression. Thus, Lasso automatically selects more relevant features
                                # and discards the others, whereas Ridge regression never fully discards any features. Some feature selection techniques are developed based on the LASSO including
                                # Bolasso which bootstraps samples,[12] and FeaLect which analyzes the regression coefficients corresponding to different values of \alpha to score all the features.

                                dual = True, # Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
                                random_state = 0 # The seed of the pseudo random number generator to use when shuffling the data.
                            ),
                        grid_values,
                        scoring = 'roc_auc', # A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
                        cv = 20 # If an integer is passed, it is the number of folds.
                       )
# Try to set the scoring on what the contest is asking for.
# The contest says scoring is for area under the ROC curve, so use this.

# Fit the classifier according to the given training data.
modelLR.fit(trainAfterFit,predCol)

print(modelLR.cv_results_)
'''
Contains scores for all parameter combinations in param_grid. Each entry corresponds to one parameter setting. Each named tuple has the attributes:
parameters, a dict of parameter settings
mean_validation_score, the mean score over the cross-validation folds
cv_validation_scores, the list of scores for each fold
'''
# Make prediction on fitted test data. These are Probability estimates. The returned estimates for all classes are ordered by the label of classes.
LRresult = modelLR.predict_proba(testAfterFit)[:,1]
# Create and store predictions in DataFrame and csv
LRoutput = pd.DataFrame(data={"id":test["id"], "sentiment":LRresult})
LRoutput.to_csv('../results/LogRegPredictions.csv', index=False, quoting=3)

def pred(s):
    return modelLR.predict(vz.vect.preprocess(s))
if __name__ == '__main__':
    main()