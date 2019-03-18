import numpy as np
import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import cross_val_score

def pred():
    # Load fitted training data
    trainAfterFit = pickle.load(open("../picks/fittedTrainData.pkl","rb"))
    # Load prediction column
    predCol = pickle.load(open("../picks/predCol","rb"))
    # Load fitted test data
    testAfterFit = pickle.load(open("../picks/fittedTestData.pkl","rb"))
    # Load test data
    test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3 )
    # Initialize MNB Classifier
    modelMNB = MNB()
    # Fit the classifier according to the given training data.
    modelMNB.fit(trainAfterFit,predCol)
    # Display stats for MB Classifier. This will give us a 20-fold cross validation score that looks at ROC_AUC 
    print ("20 Fold CV Score for Multinomial Naive Bayes: ", np.mean(cross_val_score(modelMNB, trainAfterFit, predCol, cv=20, scoring='roc_auc')))
    # Make prediction on fitted test data. These are Probability estimates. The returned estimates for all classes are ordered by the label of classes.
    MNBresult = modelMNB.predict_proba(testAfterFit)[:,1]
    # Create and store predictions in DataFrame and csv
    MNBoutput = pd.DataFrame(data={"id":test["id"], "sentiment":MNBresult})
    MNBoutput.to_csv('../results/MNBPredictions.csv', index = False, quoting = 3)

# if __name__ == '__main__':
#     main()