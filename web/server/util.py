import Tweet
import yf
import predict
import math
import pandas as pd



def get_price(i):
    dataframe1 = pd.read_csv(r'nifty50.csv')
    #print(dataframe1)
    i=int(i)
    x=dataframe1['NAME'].tolist()
    y=dataframe1['SYMBOL'].tolist()
    z=dataframe1['Shortforms'].tolist()

    name = x[i]
    symbol = y[i]
    sf = z[i]
    company = f'\"{x[i]}\"'
    for j in z[i].split(","):
        company = company + " OR " + f'\"{j}\"'



    yf.Stocks(symbol)
    Tweet.Twitter(name,company,i)
    a = predict.Predict(name,symbol)
    a = a.item()
    a = round(a,2)
    return a

#get_price("i")

    
    
    















    
     
