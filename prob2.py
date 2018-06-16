import sklearn
import numpy as np
import pandas as pd       
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.utils import shuffle


def parseDataset(feature):

    print "Parsing training data"
    data = pd.read_csv("tweets/semeval/reg_trainSet.tsv", sep='\t')
    Y = data["class"]
    tweets = data["tweet"]
    
    print "Parsing test data"
    testdata = pd.read_csv("tweets/semeval/testSet.tsv", sep='\t')
    Y_test = testdata["class"]
    tweets_test = testdata["tweet"] 
    
    print "Vectorizing ..."
    if feature == "1":
        tfidf_vect = TfidfVectorizer(min_df = 1, stop_words=None, ngram_range=(1,2), use_idf=True)
        X = tfidf_vect.fit_transform(tweets)
        X_test = tfidf_vect.transform(tweets_test)
        print X.shape
    elif feature == "2":
        count_vect = CountVectorizer(min_df = 1, binary=True, stop_words=None, ngram_range=(1,2))
        X = count_vect.fit_transform(tweets)            
        X_test = count_vect.transform(tweets_test)
        print X.shape
    return X, Y, X_test, Y_test


def classifier(X, Y, X_test, Y_test, opt):

    if opt == '1':
        LR = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, class_weight=None, 
                            solver='liblinear', max_iter=100, 
                            multi_class='ovr', verbose=3, warm_start=False, n_jobs=1)
    elif opt == '2':
        LR = MultinomialNB()
                 
    LR.fit(X,Y)  
    
    
    Y_pred = LR.predict(X_test)
    y_probas = LR.predict_proba(X_test)
    print type(y_probas)
    skplt.metrics.plot_precision_recall_curve(Y_test, y_probas, curves=['each_class'])
    plt.show()
    skplt.metrics.plot_precision_recall_curve(Y_test, y_probas, curves=['micro'])
    plt.show()
    skplt.estimators.plot_learning_curve(LR, X, Y)
    plt.show()
    print metrics.classification_report(Y_test, Y_pred)

    
if __name__ == '__main__':

    print "===================== SELECT CLASSIFIER ======================================"
    print "LOGISTIC REGRESSION     ---> PRESS 1"
    print "NAIVE BAYES             ---> PRESS 2"
    print ""
    opt = raw_input("1/2: ")
    print "===================== FEATURE TYPE ======================================"
    print "USE TF-IDF FEATURES      ---> PRESS 1"
    print "USE BOOLEAN FEATURES     ---> PRESS 2"
    feature = raw_input("1/2: ")
    
    X, Y, X_test, Y_test = parseDataset(feature)
    classifier(X, Y, X_test, Y_test, opt)


