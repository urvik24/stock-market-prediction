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

    time= datetime.datetime.now()
    time1 = time - timedelta (hours=5.5)


    
    start_time = datetime.datetime(2021,1,1,00,00,00)
    date_time1 = start_time.strftime("%Y-%m-%d %H:%M:%S")


    c.Since = date_time1
    #c.Until = date_time

    print("Fetching the tweets for",company)

    twint.run.Search(c)
    tweets = twint.storage.panda.Tweets_df
    
    print(tweets.shape)
    print("\nFetched all tweets and imported!")
    tweets.to_csv(f"{name}.csv", index = False, header=True)

#Twitter("Cipla","Cipla",9)

#Twitter("name","Cipla",9)
