import warnings
warnings.filterwarnings('ignore')
import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Dense, Activation

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata

def Train(name,symbol):
    # reading the datasets
    stock_price = pd.read_csv(f"{symbol}.csv")
    stock_headlines = pd.read_csv(f"{name}.csv")

    # check the null values in datasets
    stock_price.isna().any(), stock_headlines.isna().any()
    # removing duplicates
    stock_price = stock_price.drop_duplicates()
    stock_headlines = stock_headlines.drop_duplicates()
    # coverting the datatype of column 'Date' from object to 'datetime'
    stock_price['Date'] = pd.to_datetime(stock_price['Date']).dt.normalize()
    stock_headlines['date'] = pd.to_datetime(stock_headlines['date']).dt.normalize()
    # Filtering
    stock_price = stock_price.filter(['Date', 'Close', 'Open', 'High', 'Low', 'Volume'])
    stock_headlines = stock_headlines.filter(['date', 'tweet'])
    # 'Date' as the index column
    stock_price.set_index('Date', inplace= True)
    # sorting the data according to the index i.e 'Date'
    stock_price = stock_price.sort_index(ascending=True, axis=0)
        
    cp = stock_price['Close'].tolist()
    previous_cp = cp[-1]

    # grouping the news headlines according to 'Date'
    stock_headlines = stock_headlines.groupby(['date'])['tweet'].apply(lambda x: ','.join(x)).reset_index()

    # setting column 'Date' as the index column
    stock_headlines.set_index('date', inplace= True)

    # sorting the data according to the index i.e 'Date'
    stock_headlines = stock_headlines.sort_index(ascending=True, axis=0)

    # concatenating the datasets stock_price and stock_headlines
    stock_data = pd.concat([stock_price, stock_headlines], axis=1)
    stock_data.to_csv('stp.csv')
         
    # dropping the null values if any
    stock_data.dropna(axis=0, inplace=True)


    # adding empty sentiment columns to stock_data for later calculation
    stock_data['compound'] = ''
    stock_data['negative'] = ''
    stock_data['neutral'] = ''
    stock_data['positive'] = ''

    # instantiating the Sentiment Analyzer
    sid = SentimentIntensityAnalyzer()

    # calculating sentiment scores
    stock_data['compound'] = stock_data['tweet'].apply(lambda x: sid.polarity_scores(x)['compound'])
    stock_data['negative'] = stock_data['tweet'].apply(lambda x: sid.polarity_scores(x)['neg'])
    stock_data['neutral'] = stock_data['tweet'].apply(lambda x: sid.polarity_scores(x)['neu'])
    stock_data['positive'] = stock_data['tweet'].apply(lambda x: sid.polarity_scores(x)['pos'])


    # dropping the 'headline_text' which is unwanted now
    stock_data.drop(['tweet'], inplace=True, axis=1)

    # rearranging the columns of the whole stock_data
    stock_data = stock_data[['Close', 'compound', 'negative', 'neutral', 'positive', 'Open', 'High', 'Low', 'Volume']]


    # writing the prepared stock_data to disk
    stock_data.to_csv('stock_data.csv')

    # re-reading the stock_data into pandas dataframe
    stock_data = pd.read_csv('stock_data.csv', index_col = False)

    # renaming the column
    stock_data.rename(columns={'Unnamed: 0':'Date'}, inplace = True)

    # setting the column 'Date' as the index column
    stock_data.set_index('Date', inplace=True)

    # checking for null values
    stock_data.isna().any()

    # calculating data_to_use
    percentage_of_data = 1.0
    data_to_use = int(percentage_of_data*(len(stock_data)-1))

    # using data for training
    train_end = int(data_to_use*1.0)
    total_data = len(stock_data)
    start = total_data - data_to_use
    # predicting one step ahead
    steps_to_predict = 1

    # capturing data to be used for each column
    close_price = stock_data.iloc[start:total_data,0] #close
    compound = stock_data.iloc[start:total_data,1] #compound
    negative = stock_data.iloc[start:total_data,2] #neg
    neutral = stock_data.iloc[start:total_data,3] #neu
    positive = stock_data.iloc[start:total_data,4] #pos
    open_price = stock_data.iloc[start:total_data,5] #open
    high = stock_data.iloc[start:total_data,6] #high
    low = stock_data.iloc[start:total_data,7] #low
    volume = stock_data.iloc[start:total_data,8] #volume

    # printing close price
    #print("Close Price:")
    #print(close_price)


    # shifting next day close
    close_price_shifted = close_price.shift(-1) 

    # shifting next day compound
    compound_shifted = compound.shift(-1) 

    # concatenating the captured training data into a dataframe
    data = pd.concat([close_price, close_price_shifted, compound, compound_shifted, volume, open_price, high, low], axis=1)

    # setting column names of the revised stock data
    data.columns = ['close_price', 'close_price_shifted', 'compound', 'compound_shifted','volume', 'open_price', 'high', 'low']

    # dropping nulls
    data = data.dropna()

    # setting the target variable as the shifted close_price
    y = data['close_price_shifted']

    # setting the features dataset for prediction  
    cols = ['close_price', 'compound', 'compound_shifted', 'volume', 'open_price', 'high', 'low']
    x = data[cols]
    #print(x)

    # scaling the feature dataset
    scaler_x = preprocessing.MinMaxScaler (feature_range=(-1, 1))
    x = np.array(x).reshape((len(x) ,len(cols)))
    x = scaler_x.fit_transform(x)

    # scaling the target variable
    scaler_y = preprocessing.MinMaxScaler (feature_range=(-1, 1))
    y = np.array (y).reshape ((len( y), 1))
    y = scaler_y.fit_transform (y)

    # reshaping the feature dataset for feeding into the model
    x = x.reshape (x.shape + (1,))


    # setting the seed to achieve consistent and less random predictions at each execution
    np.random.seed(2016)

    # setting the model architecture
    model=Sequential()
    model.add(LSTM(100,return_sequences=True,activation='tanh',input_shape=(len(cols),1)))
    model.add(Dropout(0.1))
    model.add(LSTM(100,return_sequences=True,activation='tanh'))
    model.add(Dropout(0.1))
    model.add(LSTM(100,activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    # printing the model summary
    #model.summary()

    # compiling the model
    model.compile(loss='mse' , optimizer='adam')

    # fitting the model using the training dataset
    model.fit(x, y, validation_split=0.2, epochs=500, batch_size=64, verbose=1)

    model.save(f'{name}')
    print("Model saved")

#Train("Infosys","INFY.NS")

    

