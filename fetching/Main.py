import Tweet
import yf
import training
import math
import pandas as pd

dataframe1 = pd.read_csv(r'nifty50.csv')
print(dataframe1)
x=dataframe1['NAME'].tolist()
y=dataframe1['SYMBOL'].tolist()
z=dataframe1['Shortforms'].tolist()


i=int(input("Select the number of the company: "))
print("The company selected is",x[i])

#print(a)
name = x[i]
symbol = y[i]
sf = z[i]
company = f'\"{x[i]}\"'
for j in z[i].split(","):
    company = company + " OR " + f'\"{j}\"'



yf.Stocks(symbol)
Tweet.Twitter(name,company,i)
training.Train(name,symbol)

