
# coding: utf-8

# In[ ]:

import pandas as pd
from pandas_datareader import data
import datetime
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import requests
from bs4 import BeautifulSoup
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


stocks = {
'Pfizer' : 'PFE',
'United Technologies' : 'UTX',
'Walt Disney' : 'DIS',
'Visa' : 'V',
'Boeing' : 'BA',
'Procter & Gamble' : 'PG',
'Johnson & Johnson' : 'JNJ',
'Coca Cola' : 'KO',
'UnitedHealth Group' : 'UNH',
'McDonalds Corporation' : 'MCD',
'Walmart' : 'WMT',
'NIKE' : 'NKE',
'Microsoft' : 'MSFT',
'Home Depot' : 'HD',
 'Apple' : 'AAPL',
'3M' : 'MMM',
'Cisco' : 'CSCO',
'Verizon Communications' : 'VZ',
'IBM' : 'IBM',
'Merck & Co.' : 'MRK',
'General Electrics' : 'GE',
'Travelers' : 'TRV',
'Intel' : 'INTC',
'Goldman Sachs Group' : 'GS',
'American Express' : 'AXP',
'JPMorgan' : 'JPM',
'DD' : 'DD',
'Exxon Mobil' : 'XOM',
'Caterpillar' : 'CAT',
'Chevron Corporation' : 'CVX',
'CISCO' : 'CSCO',
 'Apex' : 'APEX'
}

FMCG = []
IT = ['IBM','Intel','Apple','Microsoft','Cisco']
Banking_Finance = ['American Express','Goldman Sachs Group','JPMorgan','Travelers','Visa' ]
Pharma_HealthServices = ['Pfizer','DD','Johnson & Johnson','Merck & Co.','UnitedHealth Group']
Automobile = []
Retail = ['WalMart','Home Depot']
Production_Manufacturing = ['3M','Caterpillar','General Electric','United Technologies']
Aerospace_Defence = ['Boeing']
Oil_Gas = ['Chevron Corporation','Exxon Mobil']
Consumer_NonDurables = ['Coca Cola','NIKE','Procter & Gamble']
Consumer_Services = ['Walt Disney','McDonalds Corporation']
Telecommunication = ['Verizon Communications']

def crawl (stock_name):
 u = "http://timesofindia.indiatimes.com/topic/"+stock_name
 source = requests.get(u)
 txt = source.text
 soup = BeautifulSoup(txt)
 content = soup.findAll('meta', itemprop='name')
 s = str((content[1]))
 news1 = s.split('"')[1] 
 print(news1)
 return news1

def extract_features(word_list):
    return dict([(word, True) for word in word_list])

def sentiment_analysis(news):
    # Load positive and negative reviews
    print("Sentiment Analsysis", news)
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')

    features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
            'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
            'Negative') for f in negative_fileids]

    # Split the data into train and test (80/20)
    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))

    features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
    print('\nNumber of training datapoints:', len(features_train))
    print('Number of test datapoints:', len(features_test))

    # Train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(features_train)
    print('\nAccuracy of the classifier:', nltk.classify.util.accuracy(classifier, features_test))

    #print('\nTop 10 most informative words:')
    #for item in classifier.most_informative_features()[:10]:
        #print(item[0])

    input_reviews = []
    input_reviews.append(news)

    print('\nPredictions:')
    for review in input_reviews:
        print('\nReview:', review)
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        print('Predicted sentiment:', pred_sentiment)
        print('Probability:', round(probdist.prob(pred_sentiment), 2))


def listing() :

    li=stocks.keys()
    for i in li :
        print(i)


def analyze() :

    stk = str(input("Which stock do you want me to analyze?"))
    s = stocks.get(stk)
    if s == 0 :
        print("Please enter the correct keyword : Refer to this list ")
        print(stocks.keys)
    else :   
        print("Enter the corresponding number")
        print("1 for graphical analysis")
        print("2 to get stock details")
        print("3 for sentiment analysis")
        print("4 to go back")
        inp_a = int(input())
        if inp_a==1 :
            graph(s)
        elif inp_a==2 :
            listing()
        elif inp_a==3 :
            news = crawl(stk)
            sentiment_analysis(news)
        

def prediction():
    count=1
    stck = str(input("Which stock do you want to predict"))
    pre = stocks.get(stck)
    if pre :
        print("Enter the starting date for the data you want to use for prediction")
        sy=int(input("Year : "))
        sm=int(input("Month : "))
        sd=int(input("Date : "))
        start = datetime(sy,sm,sd)
        end = datetime.today()
        df = data.DataReader(pre, 'google', start, end)
        print(df.tail())
        Y = np.array(df['Close'])
        X = np.array(range(1,len(df.index)+1))
        Z = Y[-1:-11:-1]
        fpred = 0
        news = crawl(stck)
        sentiment_analysis(news)

        while not fpred :
            p = np.poly1d(np.polyfit(X,Y,count)) 
            pred = p(len(df.index)+1)     
            diff = max(Z)-min(Z)
            if pred<=Z[0]+diff and pred>=Z[0]-diff :
                fpred = pred
            else :
                count+=1
               
        r2=r2_score(Y,p(X))
        print("\n\nr2 score -> " , r2)
        df_last=df.tail(1)
        df_last=df_last['Close']
        print("Current Price -> ", df_last)
        print("Predicted Price for tomorrow -> ",fpred)
        print("At -> ",len(df.index)+1)
        print("Degree of Regression Model -> ",count)
        plt.scatter(X,Y)
        plt.plot(X, p(X), c='r')
        plt.ylabel('Price($)')
        plt.xlabel('Days')
        plt.show()
        
    else : 
        print("Please enter the correct keyword : Refer to this list ")
        lis=stocks.keys()
        for i in lis :
         print(i)

def graph(stock_name):      
        print("Choose an option to generate the graph :")
        print("Enter 1 for graph based on last day data")
        print("Enter 2 for graph based on 1 week data")
        print("Enter 3 for graph based on 1 month data")
        print("Enter 4 for graph based on 1 year data")
        print("Enter 5 to go back")
        inp_g = int(input())
        if inp_g==1 :            
                end1 = datetime.today()-timedelta(60)
                df1 = data.DataReader(stock_name, 'google', end1)
                df2 = df1.tail(2)
                Y1 = np.array(df2['Close'])
                var = ((Y1[-1]-Y1[0])/Y1[-1])*100
                X1 = np.array(range(1,len(df2.index)+1))
                plt.plot(X1,Y1,c='b')
                
                plt.show()
                print("Change ",var,"%")
        elif inp_g==2 :            
                end1 = datetime.today()-timedelta(60)
                df1 = data.DataReader(stock_name, 'google', end1)
                df2 = df1.tail(7)
                Y1 = np.array(df2['Close'])
                var = ((Y1[-1]-Y1[0])/Y1[-1])*100
                X1 = np.array(range(1,len(df2.index)+1))
                plt.plot(X1,Y1,c='b')
                plt.show()
                print("Change ",var,"%")
        elif inp_g==3 :            
                end1 = datetime.today()-timedelta(60)
                df1 = data.DataReader(stock_name, 'google', end1)
                df2 = df1.tail(30)
                Y1 = np.array(df2['Close'])
                var = ((Y1[-1]-Y1[0])/Y1[-1])*100
                X1 = np.array(range(1,len(df2.index)+1))
                plt.plot(X1,Y1,c='b')
                plt.show()
                print("Change ",var,"%")
        elif inp_g==4 :            
                end1 = datetime.today()-timedelta(550)
                df1 = data.DataReader(stock_name, 'google', end1)
                df2 = df1.tail(365)
                Y1 = np.array(df2['Close'])
                var = ((Y1[-1]-Y1[0])/Y1[-1])*100
                X1 = np.array(range(1,len(df2.index)+1))
                plt.plot(X1,Y1,c='b')
                plt.show()
                print("Change ",var,"%")
                
        
        
print("---------------------MENU--------------------")
print("1 . List all the stocks")
print("2 . Analysis and Suggestion")
print("3 . Predict a stock")
print("4 . Exit")
inp = int(input("Enter your choice"))
if inp == 1 :
 listing()
elif inp == 2 :
 analyze()
elif inp == 3 :
 prediction()
else :
 print("Wrong input")



# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



