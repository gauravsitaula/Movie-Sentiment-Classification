# Kaggle-IMDB Data Sentiment Analysis

Here I am trying to solve the sentiment analysis problem for movie reviews. The problem is taken from the Kaggle competition 

https://www.kaggle.com/c/word2vec-nlp-tutorial

I will be using python as my programming language. For this, I have used the Anaconda 2.7 package.

I have used three different classifiers to solve this problem. All of the classifiers have a common pre processing step where I perform data cleanup and then use TfidfVectorizer for feature selection

##Instructions to run

1. Clone this git repo to a suitable location on your machine.

2. Download and unzip the following data files

	testData.tsv 	
	labeledTrainData.tsv
	
	from https://www.kaggle.com/c/word2vec-nlp-tutorial/data and store them in the `data` folder

3. Run the `classify.py` script in the `imdbMain` package. This will make predictions as per all three algorithms.

4. Once the script has terminated, the final predictions should be in the `results` folder

##Explanation

Here is a description of the components

1.  `classify.py` in the `imdbMain` package

    This is the driver script. It runs the code for feature selection and classification.

2.  `vectorize.py` in the `util` package

    This script is responsible for feature selection using `TfidfVectorizer`. Please look at the well documented script to understand the code. Here are the main steps
  
	1. Load training and test data
	2. Data cleanup by calling the  `sentimentToWordlist` function of `preProc.py`script in the `cleanup` package
	3. Initialize `TfidfVectorizer` with appropriate parameters for feature selection
	4. Fit the cleaned up training and test data on the `TfidfVectorizer` created above
	5. Store the fitted data
    
3. `preProc.py`in the `cleanup` package

    This script is responsible for cleaning up the data and making it suitable for feature selection. It has a function `sentimentToWordlist` that takes a raw movie review as input and performs the following steps
    
    1. Use BeautifulSoup library to remove the HTML/XML tags (e.g., `<br/>`)
    2. Check and remove/keep smileys, numbers and stopwords as indicated by the various flags
    3. Convert all text to lowercase
    4. Return a list of words
 
4. `predict`package
    This package contains three scripts, one for eac of the three classifiers. The steps taken in all of them are

	1. Load the training and test data fitted on `TfidfVectorizer`
	2. Initialize the classifier
	3. Fit the classifier on the  training data
	4. Make predictions on the test data
	5. Store prediction in the `results` folder

