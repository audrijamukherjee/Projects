import numpy as np
import pandas as pd
import datetime
import re
import nltk
import time
np.seterr(divide='ignore', invalid='ignore')
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve, auc,accuracy_score
from sklearn.model_selection import ShuffleSplit
from newspaper import Article
import re
from nltk.corpus import stopwords
from sklearn.externals import joblib


class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        review_text=review
        #
        # 1. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 2. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 3. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)


real_all= np.load("Real_clean.npy")
fake_all=np.load("Fake_clean.npy")
before_clean_real=np.load("before_clean_real.npy")
before_clean_fake=np.load("before_clean_fake.npy")
total_real=real_all.shape[0]
total_fake=fake_all.shape[0]
total=total_real+total_fake
#join the real and fake samples


all_news=np.concatenate((real_all,fake_all),axis=0)
unclean_news=np.concatenate((before_clean_real,before_clean_fake),axis=0)
#UMCOMMENT TO TRAIN, run make_predictions

#add labels
#Real=0
#Fake=1
y_labels=np.concatenate((np.zeros(total_real), np.ones(total_fake)), axis=0)

#mix them up
indices=np.arange(total)
np.random.shuffle(indices)
all_news=all_news[indices]  #entire training+test data of news content
unclean_news=unclean_news[indices]
#unclean_news[:,1]=unclean_news[:,1][indices]
y_labels=y_labels[indices]


# ****** Create a bag of words from the training set
# Initialize the "CountVectorizer" object, aka bag of words.
#calculate X_counts and X_tfidf
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,stop_words = None,max_features = 5000)
X_counts = vectorizer.fit_transform(all_news)
X_counts = X_counts.toarray()

# calculating X_tfidf- modifies  X_counts  by applying the sklearn tfidf vectorizer
transformer = TfidfTransformer(smooth_idf=False)
X_tfidf = transformer.fit_transform(X_counts)

#split into train and test sets
X_tfidf_train=X_tfidf[0:-500]
X_tfidf_test=X_tfidf[-500:]
Y_train=y_labels[0:-500]
Y_test=y_labels[-500:]

X_tfidf_train = csr_matrix(X_tfidf_train)
X_tfidf_test = csr_matrix(X_tfidf_test)

np.save("all_news.npy",all_news)
np.save("unclean_news.npy",unclean_news)
np.save("indices.npy",indices)
np.save("X_tfidf_train.npy",X_tfidf_train)
np.save("X_tfidf_test.npy",X_tfidf_test)
np.save("Y_train.npy",Y_train)
np.save("Y_test.npy",Y_test)
'''
def tune_SVM(X_train, X_test, y_train, y_test,iterations=30):
    scores_C=[]
    C_list=[]
    random.seed(0)

    for i in range(0,iterations):
        #generating a random value for the exponent of 10.
        #This gives us a uniform random value generation over 1e-4 to 1e+4
        C_curr=10**random.uniform(-4,4)
        estimator = LinearSVC(C=C_curr,random_state=0)
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=5, scoring='roc_auc')
        C_list.append(C_curr)
        scores_C.append(np.mean(test_scores))

    #keep the value of C giving maximum roc_auc score
    best_C= C_list[np.argmax(scores_C)]
    print ('Lift between maximum and minimum ROC-AUC = ',np.max(scores_C)-np.min(scores_C))
    return best_C, np.max(scores_C)

print 'Training....'
# tuning C for SVM for the X_tfidf using only training data
####
XT_train, XT_test, yT_train, yT_test = train_test_split(X_tfidf_train, Y_train, test_size=0.2,\
                                                        random_state=0)
X_tfidf_C, X_tfidf_AUC = tune_SVM(XT_train, XT_test, yT_train, yT_test, 30)

print " Best C= ",X_tfidf_C," AUC= ", X_tfidf_AUC
#X_tfidf_C=0.297423091594
# Retrain the X_tfidf classifier on the 80% train data for X_tfidf
#X_tfidf_estimator = LinearSVC(C=X_tfidf_C, random_state=0).fit(XT_train,yT_train)  ##change


# Retrain the X_tfidf classifier on the whole data for X_tfidf_train
X_tfidf_estimator = LinearSVC(C=X_tfidf_C, random_state=0).fit(X_tfidf_train,np.array(Y_train))  ##change

#save the model
joblib.dump(X_tfidf_estimator, 'LinearSVC.pkl')

#predict for test data
def make_prediction_LinSVC(test_X, estimator, X_type):
    #X_type is for me to know which X matrix we are talking about
    #predict fakeness for test set-test_X
    predicted_test=estimator.predict(test_X)

    # Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
    output = np.array(predicted_test)
    np.save("Output_predicted.npy",output)
    # Use pandas to write the comma-separated output file
    output.to_csv('./Output_'+X_type+'.csv', index=False, quoting=1)
    print ("Wrote results to Output_predicted.npy", time.ctime())
    return output

print "predicting....."
Y_pred=make_prediction_LinSVC(X_tfidf_test,X_tfidf_estimator,"tfidf")
#calc accuracy
print "Accuracy: ",accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)


#because we need confidence score for the URL also, using SVC
X_tfidf_estimator = SVC(kernel='linear',probability=True).fit(X_tfidf_train,np.array(Y_train))  ##change
#trained


'''


###make predictions for the SVC model
#because we need confidence score for the URL also, using SVC
#X_tfidf_train=np.load("X_tfidf_train.npy")
#X_tfidf_test=np.load("X_tfidf_test.npy")
#Y_train=np.load("Y_train.npy")
#Y_test=np.load("Y_test.npy")
print X_tfidf_train.shape,Y_train.shape,X_tfidf_test.shape,Y_test.shape
X_tfidf_estimator = SVC(kernel='linear',probability=True).fit(X_tfidf_train,Y_train)
#save the model
joblib.dump(X_tfidf_estimator, 'SVC.pkl')

#predict for test data
def make_prediction_SVC(test_X, estimator, X_type):
    #X_type is for me to know which X matrix we are talking about
    #predict fakeness for test set-test_X
    predicted_test=estimator.predict(test_X)
    confidence = estimator.predict_proba(test_X)
    # print "Prediction=", prediction
    # print "Confidence=",confidence
    # Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
    output = np.array(predicted_test)
    np.save("Output_predicted.npy",output)
    #save confidence scores
    output_confidence=np.array(confidence)[:,1]
    np.save("Output_confidence.npy",output_confidence)
    return output

print "predicting....."
Y_pred=make_prediction_SVC(X_tfidf_test,X_tfidf_estimator,"tfidf")
#calc accuracy
print "Accuracy: ",accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)


###This is for LinearSVC 0/1 score
def query_fakeness1(url):
    X_tfidf_estimator = joblib.load('LinearSVC.pkl')

    #print "URL received: ", url
    clean_query = []
    if url == None:
        return -1
    try:
        article = Article(url)
        article.download()
        article.parse()
        content = article.text
        if content != None or content != np.nan and 'jp.wsj' not in url:  # this one URL gives garbage
            clean_query.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(content, True)))
            # print row

    except:
        return -1
    # all_news = np.concatenate((clean_query,real_all, fake_all), axis=0)
    all_news[0] = clean_query[0]
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    X_counts = vectorizer.fit_transform(all_news)
    X_counts = X_counts.toarray()

    # calculating X_tfidf- modifies  X_counts  by applying the sklearn tfidf vectorizer
    transformer = TfidfTransformer(smooth_idf=False)
    X_tfidf = transformer.fit_transform(X_counts)

    X_tfidf_query = csr_matrix(X_tfidf)
    prediction = X_tfidf_estimator.predict(X_tfidf_query[0])
    #confidence = X_tfidf_estimator.predict_proba(X_tfidf_query[0])
    #print "Prediction=", prediction
    #print "Confidence=", confidence
    #return confidence[0][1]  ##probability of the sample being fake news




def query_fakeness(url):
    print "URL received: ",datetime.datetime.now().time()
    X_tfidf_estimator = joblib.load('SVC.pkl')
    #print "URL received: ",url
    clean_query=[]
    if url==None:
        return -1
    try:
        article = Article(url)
        article.download()
        article.parse()
        content=article.text
        if content!=None or content!=np.nan and 'jp.wsj' not in url:  #this one URL gives garbage
            clean_query.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(content, True)))
        #print row

    except:
        return -1
    #all_news = np.concatenate((clean_query,real_all, fake_all), axis=0)
    all_news[0]=clean_query[0]
    #print "Parsed & Cleaned: ", datetime.datetime.now().time()
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    X_counts = vectorizer.fit_transform(all_news)
    X_counts = X_counts.toarray()

    # calculating X_tfidf- modifies  X_counts  by applying the sklearn tfidf vectorizer
    transformer = TfidfTransformer(smooth_idf=False)
    X_tfidf = transformer.fit_transform(X_counts)

    X_tfidf_query = csr_matrix(X_tfidf)
    #print "Converted to tfidf: ", datetime.datetime.now().time()
    prediction=X_tfidf_estimator.predict(X_tfidf_query[0])
    confidence=X_tfidf_estimator.predict_proba(X_tfidf_query[0])
    #print "Prediction=", prediction
    #print "Confidence=",confidence
    #print "Predicted: ", datetime.datetime.now().time()

    return confidence[0][1]   ##probability of the sample being fake news

'''
#In sample Real true and predicted
print query_fakeness('http://www.wsj.com/articles/syria-strike-sends-russia-a-signal-1491608375')
#In sample Real true and predicted
print query_fakeness('http://www.wsj.com/articles/ayaan-hirsi-ali-islams-most-eloquent-apostate-1491590469?mod=trending_now_5')
# New sample Fake true and predicted
print query_fakeness('http://100percentfedup.com/flashback-black-panther-diva-beyonce-at-white-house-easter-egg-roll-in-playboy-bunny-style-dress/')
#In sample actually real, predicted false
print query_fakeness('http://nypost.com/2017/04/05/pepsi-is-yanking-its-controversial-protest-ad/')
#In sample Real true and predicted
print query_fakeness('http://www.chicagotribune.com/news/nationworld/ct-theresa-may-early-election-20170418-story.html')
# New sample Real true and predicted
print query_fakeness('http://www.gadgetsnow.com/tech-news/infosys-q4-profit-beats-estimates-on-key-client-wins/articleshow/58158898.cms?utm_source=toiweb&utm_medium=referral&utm_campaign=toiweb_hptopnews')


'''
import sys
print query_fakeness(sys.argv[1])  #uncomment to run for plug in just the 2 lines here

