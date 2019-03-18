import pandas as pd
from cleanup import preProc as preProc
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def vect():
    # Load labeled training data
    train = pd.read_csv('../data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)

    # The column on which we will predict
    predCol = train['sentiment']

    pickle.dump(predCol, open("../picks/predCol", "wb"))

    # List for storing cleaned up training data
    trainData = []

    # Loop counter
    numRevs = len(train['review'])

    for i in range(0, numRevs):

        if ((i + 1) % 2000 == 0):
            print("Train Review %d of %d\n" % (i + 1, numRevs))

        # Clean each review> Please look at the definition of the sentimentToWordlist function in the preproc.py script
        trainData.append(" ".join(preProc.sentimentToWordlist(train['review'][i])))

    # Load test data
    test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3)

    # List for storing cleaned up test data
    testdata = []

    # Loop counter
    numRevs = len(test['review'])

    for i in range(0, numRevs):

        if ((i + 1) % 2000 == 0):
            print("Test Review %d of %d\n" % (i + 1, numRevs))

        # Clean each review> Please look at the definition of the sentimentToWordlist function in the preproc.py script
        testdata.append(" ".join(preProc.sentimentToWordlist(test['review'][i])))

    # Define/build TfidfVectorizer
    print("Defining TFIDF Vectorizer")

    tfIdfVec = TFIV(
        min_df=10,
        # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
        max_features=1000,
        # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
        strip_accents='unicode',
        # Remove accents during the preprocessing step. 'ascii' is a fast method that only works on characters that have an direct ASCII mapping.
        # 'unicode' is a slightly slower method that works on any characters.
        analyzer='word',  # Whether the feature should be made of word or character n-grams.. Can be callable.
        token_pattern=r'\w{1,}',
        # Regular expression denoting what constitutes a "token", only used if analyzer == 'word'.
        ngram_range=(3, 3),
        # The lower and upper boundary of the range of n-values for different n-grams to be extracted.
        use_idf=1,  # Enable inverse-document-frequency reweighting.
        smooth_idf=1,  # Smooth idf weights by adding one to document frequencies.
        sublinear_tf=1,  # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
        stop_words='english'  # 'english' is currently the only supported string value.
    )

    pickle.dump(tfIdfVec, open("../picks/vectorizer.pkl", "wb"))

    combineData = trainData + testdata  # Combine both to fit the TFIDF vectorization.

    trainLen = len(trainData)

    print("Fitting")

    tfIdfVec.fit(combineData)  # Learn vocabulary and idf from training set.

    print("Transforming")

    combineData = tfIdfVec.transform(
        combineData)  # Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform).
    pickle.dump(combineData, open("../picks/transformedData.pkl", "wb"))
    print("Fitting and transforming done")

    trainAfterFit = combineData[:trainLen]  # Separate back into training and test sets.
    pickle.dump(trainAfterFit, open("../picks/fittedTrainData.pkl", "wb"))

    testAfterFit = combineData[trainLen:]
    pickle.dump(testAfterFit, open("../picks/fittedTestData.pkl", "wb"))

    def preprocess(s):
        return tfIdfVec.fit(s)


#
# if __name__ == '__main__':
#     main()
