import yfinance as yf
import pandas as pd
from datetime import date
from datetime import timedelta

def Stocks(company):
    
    date1 = date.today()
    date2 = "2021-01-01"
    company1 = company[:-3]
    data = yf.download(company, date2 , date1)
    #print(data)
    data.to_csv(f"{company}.csv")
    print("Fetched Stock Details for "+company1+" for the past 60 days")

#Stocks("RELIANCE.NS")
