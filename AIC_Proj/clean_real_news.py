import re
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

#if url contains jp.wsj, throw away the row
#if content is empty throw away the row

class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        review_text=review
        # 1. Remove HTML, already done in grabbing real data
        #review_text = BeautifulSoup(review, "html.parser").get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)

#cols=['url','authors','date','title','summary','content','keywords','label']
real = pd.read_csv("FullData_Real_News.tsv", header=0, delimiter="\t")
clean_real = []
before_clean=[]
# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for i in range(0, len(real["content"])):  ##correct
    try:
        row = real.ix[i]
        if real.ix[i][5]!=None or real.ix[i][5]!=np.nan and 'jp.wsj' not in row[0]:  #this one URL gives garbage
            clean_real.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(row[5], True)))
            before_clean.append([row[1], row[5]])
            #print len(clean_real),len(before_clean)
    except:
        print "Bad row: ",i
#np_clean_real=np.array(set(clean_real))
#np_before_clean=np.array(set(before_clean))
before_clean= np.array(before_clean)
before_clean_real=pd.DataFrame(data=before_clean,index=range(before_clean.shape[0]),columns=['URL','unclean_text'])
np.save("Real_clean.npy",clean_real)
np.save("before_clean_real.npy",before_clean)
pd.DataFrame.to_csv(before_clean_real,"before_clean_real.csv",sep=',')

print "Cleaned real data..."

before_clean_real=pd.read_csv("before_clean_real.csv",header=0,delimiter=',')
before_clean_fake=pd.read_csv("before_clean_fake.csv",header=0,delimiter=',')
before_clean=np.load("before_clean.npy")
print before_clean.shape, len(clean_real)
##TOTAL 1309


