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
fake = pd.read_csv("fake.csv", header=0, delimiter=",")

clean_fake=[]

#we need the URLs and the before word list data(original article) for the results
before_clean_fake=pd.DataFrame(data=np.zeros((1500,2)),index=range(1500),columns=['URL','unclean_text'])
# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for i in range(0, 1500):  ##Right now for 1500, to keep a balanced dataset  #CHANGE BACK
    #i=10 has some only spaces, not removed that yet
    try:
        row = fake.ix[i]
        if row[5]!=None or row[5]!='' or row[5]!=np.nan and 'jp.wsj' not in row[0]:  #this one URL gives garbage
            before_clean_fake.ix[i]=[row[8],row[5]]
            clean_fake.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(row[5], True)))  ##UNCOMMENT
    except:
        print "Bad row: ",i
np_clean_real=np.array(clean_fake)        ##UNCOMMENT
np.save("Fake_clean_all.npy",clean_fake)      ##UNCOMMENT
np.save("before_clean_fake.npy",before_clean_fake)

print "Cleaned fake data..."
##TOTAL 1309


#make wordcount and literally run the same thing as HW1.ipynb
