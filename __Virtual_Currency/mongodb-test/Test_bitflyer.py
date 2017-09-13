# coding: utf-8
# In[32]:
import urllib.request
import json
import datetime
# HTTP ヘッダの　RFC2822　日付用に
from email.utils import parsedate_to_datetime


# In[1]:
# Test: Get public info
endPoint_bf='https://api.bitflyer.jp'

def getResponse_bf(path):
    global endPoint_bf
    url=endPoint_bf+path
    response=urllib.request.urlopen(url,timeout=30)
    res_date=parsedate_to_datetime(response.headers['date'])
    content = json.loads(response.read().decode('utf8'))
    return res_date,content

def getResponse_private_bf(path):
    pass


# In[37]:
# Public なもの
# 取り扱いマーケットの取得
def getMarkets_bf():
    path_Markets='/v1/markets'
    getResponse_bf(path_Markets)
    res_date,json_data = getResponse_bf(path_Markets)
    return res_date,json_data

# マーケットごとの板(取引状況)をとる
def getBoard_bf(product_code):
    global endPoint_bf    
    path_api='/v1/board'
    query='?product_code='+product_code
    path=path_api+query
    res_date,board = getResponse_bf(path)
    return res_date,board

# マーケットごとのTickerをとる
def getTicker_bf(product_code):
    global endPoint_bf
    path='/v1/ticker'
    query='?product_code='+product_code
    url=endPoint_bf+path+query
    response=urllib.request.urlopen(url)
    board = json.loads(response.read().decode('utf8'))
    return board

'''
# Private なもの
api_key_bf='YOUR_API_KEY'

#import base64
#api_secret_bf=base64.b64decode(b'YOUR_API_SECRETTxY=')
api_secret_bf=b'YOUR_API_SECRETTxY='

# sha256 署名 hmac が必要なので暗号化ライブラリのインポート
import hmac
import hashlib


# In[7]:

# 自分の所有資産確認
def getMyBalance_bf():
    global api_key_bf, api_secret_bf, endPoint_bf
    
    timestamp = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())
    method = "GET"
    path = "/v1/me/getbalance"

    text=(timestamp + method + path).encode('utf-8')
    
    m=hmac.new(api_secret_bf,text,hashlib.sha256)
    signature=m.hexdigest()
    
    headers = {
        'ACCESS-KEY':api_key_bf,
        'ACCESS-TIMESTAMP':timestamp,
        'ACCESS-SIGN':signature
    }
    
    url=endPoint_bf+path
    
    req = urllib.request.Request(url, None, headers)
    response = urllib.request.urlopen(req)
    
    balance = json.loads(response.read().decode('utf8'))
    return balance

myBalance=getMyBalance_bf()
myBalance


# In[ ]:

# 自分の注文記録確認
def getMyHistoryOrder_bf():
    global api_key_bf, api_secret_bf, endPoint_bf
        
    timestamp = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())
    method = "GET"
    path = "/v1/me/getchildorders"

    text=(timestamp + method + path).encode('utf-8')
    
    m=hmac.new(api_secret_bf,text,hashlib.sha256)
    signature=m.hexdigest()
    
    headers = {
        'ACCESS-KEY':api_key_bf,
        'ACCESS-TIMESTAMP':timestamp,
        'ACCESS-SIGN':signature
    }
    
    url=endPoint_bf+path
    
    req = urllib.request.Request(url, None, headers)
    response = urllib.request.urlopen(req)
    
    history = json.loads(response.read().decode('utf8'))
    return history
MyHistoryOrder=getMyHistoryOrder_bf()
MyHistoryOrder


# In[21]:

# 自分のコイン入履歴記録確認
def getMyHistoryCoinIns_bf():
    global api_key_bf, api_secret_bf, endPoint_bf
        
    timestamp = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())
    method = "GET"
    path = "/v1/me/getcoinins"

    text=(timestamp + method + path).encode('utf-8')
    
    m=hmac.new(api_secret_bf,text,hashlib.sha256)
    signature=m.hexdigest()
    
    headers = {
        'ACCESS-KEY':api_key_bf,
        'ACCESS-TIMESTAMP':timestamp,
        'ACCESS-SIGN':signature
    }
    
    url=endPoint_bf+path
    
    req = urllib.request.Request(url, None, headers)
    response = urllib.request.urlopen(req)
    
    coinIns = json.loads(response.read().decode('utf8'))
    return coinIns
MyHistoryCoinIns=getMyHistoryCoinIns_bf()
MyHistoryCoinIns


# In[13]:

dir(MyHistoryCoinIns )


# In[ ]:


'''
