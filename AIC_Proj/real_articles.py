from newspaper import Article
import numpy as np
import pandas as pd
import csv

cols=['url','authors','date','title','summary','content','keywords','label']
url_list=[]

i=0
#output1.csv has all real articles right now
with open('output1.csv') as csvfile:
    reader=csv.reader(csvfile,delimiter=",")
    for url in reader:
        url_list.append(url)
df_real_news=pd.DataFrame(data=np.zeros((len(url_list),len(cols))),index=range(len(url_list)),columns=cols)
print len(url_list)
i=0
for [url] in url_list:
    if i%100==0:
        print i
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        row=[url,article.authors,article.publish_date,article.title,article.summary,article.text,article.keywords,'0']
        #print row
        df_real_news.ix[i]=row
    except:
        print "Bad row : ",i
    i+=1

#Save it into a file
df_real_news.to_csv('FullData_Real_News.tsv',sep='\t', encoding='utf-8')
print "Done!"
#REMOVE
#print df_real_news

'''
np_urls=np.genfromtxt('output1.csv',delimiter=",")
print np_urls
#print np_urls.shape[0]
#np_urls=np.genfromtxt('../Project/real_news_urls.csv',delimiter=",")
'''