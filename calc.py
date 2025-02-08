from getData import *

def calc100_mean(user):
    df=getData(user)
    ma100=df.Close.rolling(100).mean()
    return ma100

def calc200_mean(user):
    df=getData(user)
    ma200=df.Close.rolling(200).mean()
    return ma200
#print(type(calc100_mean("TSLA")))