
# coding: utf-8

# In[10]:

# Collect Data
# 1. Ticker 
# 2. Board
import Test_bitflyer as bf
import pymongo as MongoClient
import operator
import re
from datetime import datetime
from time import sleep
from pymongo import MongoClient 


# In[11]:

markets_all = bf.getMarkets_bf() 


# In[12]:

# markets_all


# In[13]:

markets_timestamp_utc=markets_all[0]
#print(market_timestamp_utc)


# In[14]:

# マーケット一覧から先物を除外:product_code最後の9文字で判断する。
r = re.compile('[0-9]{2}[A-Z]{3}[0-9]{2}')
markets= list(filter(lambda item: not r.match(item['product_code'][-9:]), markets_all[1]))


# In[15]:

# Mongo DB 
mongo_db_name='bf'
mongo_dataset_name='_ticker_board01'


dbc = MongoClient()
# Open DB
db=dbc[mongo_db_name]

# Dataset
ds=db[mongo_dataset_name]
#
#ds.create_index('_timestamp')


# In[16]:

max_age_of_data=21600 # 6 hours.



# In[ ]:

# とりあえず 500 とってみよう。500xproducts
#for i in range(0,500):
#Infinite Loop!
while True:
    #間隔を開けるため(3sec)
    sleep(3)
    for mkt in markets:
        _tb_timestamp=datetime.now()
        _pdct=mkt['product_code']
        try:
            _ticker=bf.getTicker_bf(_pdct)
            _board=bf.getBoard_bf(_pdct)
        except Exception as e:
            print ('=== エラー内容 ===')
            print ('type:' + str(type(e)))
            print ('args:' + str(e.args))
            #print ('message:' + e.message)
            #print ('e自身:' + str(e))
	    #おそらく1分待ったほうがいい。API制限に引っかかる可能性がある。
            sleep(60)
            continue
        print('TIME: {0} PRODUCT: {1}'.format(_tb_timestamp,_pdct))
        db_entry = {'_timestamp':_tb_timestamp,'_product':_pdct,'_ticker':_ticker,'_board':_board}
        ds.insert_one(db_entry)
    # Cleanup Old Data
    # 学習データ収集中はコメントアウト
    #_tstamp=datetime.now()-timedelta(seconds=max_age_of_data)
    #ds.delete_many({'_timestamp':{'$lt':_tstamp}})


# In[9]:

# ds.drop()


# In[ ]:



