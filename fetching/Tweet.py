import twint
import pandas as pd
import datetime
from datetime import timedelta

def Twitter(name,company,i):
    c = twint.Config()
    
    c.Search = company
    c.Pandas = True
    #c.Limit = 150
    c.Verified = True

    time = "2020-12-31 23:59:59"
    time1 = "2010-01-01 00:00:00"



    c.Since = time1
    c.Until = time

    print("Fetching the tweets for",company)

    twint.run.Search(c)
    tweets = twint.storage.panda.Tweets_df
    
    print(tweets.shape)
    print("\nFetched all tweets and imported!")

    tweets.to_csv(f"{name}.csv", index = False, header=True)
    

#Twitter("name","Cipla",9)
