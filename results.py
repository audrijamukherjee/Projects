import numpy as np
import pandas as pd
import csv
content=[]
urls=[]
real = pd.read_csv("FullData_Real_News.tsv", header=0, delimiter="\t")
fake=pd.read_csv("fake.csv", header=0, delimiter=",")
indices=np.load("indices.npy")
all_news=np.load("all_news.npy")
unclean_news=np.load("unclean_news.npy")

'''
for i in range(0, len(real["content"])):  ##correct
    try:
        row = real.ix[i]
        if real.ix[i][5]!=None or real.ix[i][5]!=np.nan and 'jp.wsj' not in row[0]:  #this one URL gives garbage
            content.append(row[5])
            urls.append(row[0])
    except:
        print "Bad row: ",i
for i in range(0, 1500):  ##Right now for 1500, to keep a balanced dataset
    #i=10 has some only spaces, not removed that yet
    try:
        row = fake.ix[i]
        if row[5]!=None or row[5]!='' or row[5]!=np.nan and 'jp.wsj' not in row[0]:  #this one URL gives garbage
            content.append(row[5])
            urls.append(row[8])
    except:
        print "Bad row: ",i
content=np.array(content)
content=content[indices]'''


#urls=np.array(urls)
#urls=urls[indices]
#content=np.load("all_news.npy")
y_true=np.load("Y_test.npy")
y_pred=np.load("Output_predicted.npy")
confidence=np.load("Output_confidence.npy")
df_results=pd.DataFrame(data=np.zeros((500,6)),index=range(500),columns=['URL','Content','Cleaned_Content','True_labels','Predicted_labels','Confidence'])
#df_results['URLs']=urls[-200:]
x=unclean_news[1,:]
df_results['URL']=unclean_news[:,0][-500:]
df_results['Content']=unclean_news[:,1][-500:]
df_results['Cleaned_Content']=all_news[-500:]
df_results['True_labels']=y_true
df_results['Predicted_labels']=y_pred
df_results['Confidence']=confidence
df_results.to_csv("Results.csv")
