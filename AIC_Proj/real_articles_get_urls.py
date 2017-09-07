import newspaper
import csv

papers=['http://nypost.com/','http://www.chicagotribune.com/',\
        'https://www.washingtonpost.com/','http://www.newsday.com/','http://www.nydailynews.com/','http://www.amny.com/','http://cnn.com','https://www.nytimes.com']
url_list = []
# Open File
resultFyle = open("output3.csv",'wb')

# Create Writer Object
wr = csv.writer(resultFyle)


for each in papers:
    try:
        paper=newspaper.build(each)
        i=0

        if paper.articles!=[]:  #check for articles retrived
            for article in paper.articles:
                url_list.append(article.url)
                wr.writerow([article.url])
                print article.url
        else:
            print each+" empty"
    except:
        print "Some error"

