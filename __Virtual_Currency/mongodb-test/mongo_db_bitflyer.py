# coding: utf-8

import Test_bitflyer as bf
import numpy as np
import matplotlib.dates as md
import pandas as pd

from pymongo import MongoClient
from datetime import datetime,timedelta
from matplotlib.finance import candlestick_ohlc



class mongo_db_bitflyer:
    def __init__(self, mongo_db_name=None, mongo_dataset_name=None,index_column_name=None):
        # Parameters
        #'pythonic way' to use default values.
        #  Avoiding default parameters directly assigned as member by 1. creating Local Valiable, then 2. assign.
        #  ('def' is eecutable statement. Default Value is evaluated only when the 'def' is called.)
        #  (After that, assigned default value is reused repeatedly.)
        #  This affects when this class is used multiple times in a process.
        if mongo_db_name is None:
            mongo_db_name='bf'
            self.mongo_db_name = mongo_db_name
        if mongo_dataset_name is None:
            mongo_dataset_name='_ticker_board01'
            self.mongo_dataset_name = mongo_dataset_name        
        if index_column_name is None:
            index_column_name='_timestamp'
            self.index_column_name = index_column_name
            
        # Start
        # Connect to Mongo DB Server(local)
        dbc = MongoClient()

        # Open DB
        self.db=dbc[self.mongo_db_name]

        # Dataset.
        self.ds=self.db[self.mongo_dataset_name]
        self.ds.create_index(index_column_name)        
    
    def gather_data_in_range(self,_product_code,_time_start,_time_end,_columns):
        
        arg_filter= {'$and': 
                        [
                            {'_timestamp':{'$gte':_time_start}},
                            {'_timestamp':{'$lte':_time_end}},
                            {'_product':_product_code}
                        ]
                    }
        
        arg_columns={}
        
        for col in _columns:
            arg_columns.update({col:1})
        arg_columns.update({'_id':0})
        _ret= self.ds.find(arg_filter,arg_columns).sort([('_timestamp',1)])
        return _ret

    def gather_ohlc_data(self,_product_code,_time_start,_time_end,_resample_min):

        if _time_end < _time_start:
            print('end must be later than start!')
            return None

        if (_time_end - _time_start) < timedelta(minutes=_resample_min):
            print('need more than {0} minutes in range!'.format(_resample_min))
            return None

        # 指定されたプロダクトと時間範囲の、Timestamp で昇順にソートされた Pymongo Cursor を取得。
        # 取得する要素は、{_timestamp,_ticker} の Dictionary になる。
        # ※MongoDB には先物以外の全銘柄について、 Timestamp をインデックスとして、Ticker と Board の JSON をそのまま入れている。
        _data_to_plot=self.gather_data_in_range(_product_code,_time_start,_time_end,['_timestamp','_ticker'])

        # 必要なものをListへ
        ll=[]
        for itm in _data_to_plot:
            ll.append([itm['_timestamp'],itm['_ticker']['ltp'],itm['_ticker']['volume']])
        
        # List から Pandas Dataframe へ
        pd_df=pd.DataFrame(data =ll,columns=['Timestamp', 'LTP', 'Volume'])

        # pandas で _resample_min 分足 でのリサンプリング
        # リサンプリング のために、 Timestamp を Index に設定
        pd_df=pd_df.set_index('Timestamp')
        
        # 指定した _resample_min 分足で、OHLC データをリサンプリング
        pd_df = pd_df.resample(str(_resample_min)+'Min').ohlc().ffill()
        
        # リサンプリングしたので、Index を Reset する。
        pd_df=pd_df.reset_index()
        
        # Matplotlib のために日付をFloatに変換する。
        pd_df['Timestamp']=pd_df['Timestamp'].map(md.date2num)
        
        return pd_df.values
    
    def gather_macd_data(self,_product_code,_time_start,_time_end,_resample_min,nslow,nfast,nema):
        #
        # ToDo: Parameter Check!!
        #
        
        _data_to_plot=self.gather_data_in_range(_product_code,_time_start,_time_end,['_timestamp','_ticker'])

        # 必要なものをListへ。
        ll=[]
        for itm in _data_to_plot:
            ll.append([itm['_timestamp'],itm['_ticker']['ltp']])

        # List から Pandas Dataframe へ
        pd_df=pd.DataFrame(data =ll,columns=['Timestamp', 'LTP'])

        # pandas で _resample_min 分足 でのリサンプリング
        # リサンプリング のために、 Timestamp を Index に設定
        pd_df=pd_df.set_index('Timestamp')
        
        # 指定した _resample_min 分足で、データをリサンプリング
        pd_df = pd_df.resample(str(_resample_min)+'Min').bfill()
        
        # リサンプリングしたので、Index を Reset する。
        pd_df=pd_df.reset_index()
        
        # Matplotlib のために日付をFloatに変換する。
        pd_df['Timestamp']=pd_df['Timestamp'].map(md.date2num)
        
        slow, fast, macd = self.moving_average_convergence(list(pd_df['LTP']),nslow,nfast)

        #取得した MACD から、Signal の算出。
        signal=self.moving_average(macd, nema, type='exponential')

        return list(pd_df['Timestamp']),slow, fast, macd, signal
        
    # https://matplotlib.org/examples/pylab_examples/finance_work2.html
    def moving_average(self, x, n, type='simple'):
        """
        compute an n period moving average.

        type is 'simple' | 'exponential'

        """
        x = np.asarray(x)
        if type == 'simple':
            weights = np.ones(n)
        else:
            weights = np.exp(np.linspace(-1., 0., n))

        weights /= weights.sum()

        a = np.convolve(x, weights, mode='full')[:len(x)]
        a[:n] = a[n]
        return a
    
    # https://matplotlib.org/examples/pylab_examples/finance_work2.html
    def moving_average_convergence(self, x, nslow=26, nfast=12):
        """
        compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
        return value is emaslow, emafast, macd which are len(x) arrays
        """
        emaslow = self.moving_average(x, nslow, type='exponential')
        emafast = self.moving_average(x, nfast, type='exponential')
        return emaslow, emafast, emafast - emaslow
