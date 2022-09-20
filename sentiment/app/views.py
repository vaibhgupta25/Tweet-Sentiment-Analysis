from django.shortcuts import render, redirect
from django.urls import reverse
import pickle as pkl
from nltk.tokenize import word_tokenize
import tweepy
import pandas as pd
import numpy as np


# Create your views here.
def index(request):
    return render(request,'index.html')


def load(fileName):
    file=open(fileName,'rb')
    data=pkl.load(file)
    file.close()
    return data


#Processing the input
def textprocessing(tweet,cv,lm,stopwordsList):
    
    processedTweet=[]
    for word in tweet:
        if word.lower() not in stopwordsList:
            processedTweet.append(word.lower())

    for i in range(len(processedTweet)):
        processedTweet[i]=lm.lemmatize(processedTweet[i],pos='v')
                
        processedTweet=' '.join(processedTweet)
        processedTweet=cv.transform([processedTweet])
    return processedTweet


#converting the prediction to Alphabets
def sentiments(prediction):
    if prediction[0]==0:
            return 'Irrelevant'
    elif prediction[0]==1:
            return 'Negative'
    elif prediction[0]==2:
            return 'Neutral'
    elif prediction[0]==3:
            return 'Positive'


def accessingApi():
    api_key='q82kHNp48nqET4hYyPtYbrAtw'
    api_secret_key='yd4EGn9AkYDVDcIS4pa6N4hoFdUxhfv0vBBtVqGBvtQfY9tvSQ'
    access_token='1557008169511317510-7tS7riRguTvGGqgg5dz5IvKgTm3E7L'
    access_secret_token='UsxavyrJg14MfHFFIBvj41v4k9Osk6dFhduVfwwktCtel'

    # authentication
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_secret_token)

    return tweepy.API(auth)

def creatingDataframe(tweets):
    columns = ['Tweets']
    data = []

    for tweet in tweets:
        data.append([tweet.full_text])

    return pd.DataFrame(data, columns=columns)


def DataframeProcessing(df,cv,lm,stopwordsList):
    df.drop(['Unnamed: 0'],axis=1,inplace = True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    
    tokens=[]
    for i in  range(len(df)):
        tokens.append(word_tokenize(df[i]))
    
    wordList=[]
    for token in tokens:
        words=[]
        for word in token:
            if word not in stopwordsList:
                words.append(word)
        wordList.append(words)

    for i in range(wordList):
        for j in range(len(wordList[i])):
            wordList[i][j]=lm.lemmatize(wordList[i],[j])

    processedText=np.asarray(wordList)
    for i in range(len(processedText)):
        processedText[i]=" ".join(processedText[i])

    
        


def predict(request):

    #loading files
    cv=load('cv.pkl')
    lm=load('lm.pkl')
    stopwordsList=load('stopwordsList.pkl')
    model=load('model.pkl')
    

    #getting users input
    tweet=request.GET['tweets']

    
    #checking whether the input empty or not
    if(len(tweet)==0):
        return render(request,'project.html', {'predict':'Please Enter Something'})


    #User typing their own tweet
    if(tweet[0]!= '#' and tweet[0]!="@"):

        tweet=word_tokenize(tweet.lower())
        processedText=textprocessing(tweet,cv,lm,stopwordsList)
        prediction = model.predict(processedText)

        msg=sentiments(prediction)
        return render (request,'project.html', {'note':"Prediction is : ",'predict':msg})
    
    api=accessingApi()
    
    if(tweet[0]=='@'):
        limit=300

        tweets = tweepy.Cursor(api.user_timeline, screen_name=tweet, count=200, tweet_mode='extended').items(limit)
        df=creatingDataframe(tweets)
        df.to_csv('tweet.csv')
        df=df.to_html
    return render (request,'project.html', {'note':"Prediction is : ",'predict':df})

    
     

