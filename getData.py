import yfinance as yf
import datetime as dt
def getData(ticker):
    start = "2010-01-01"
    end = dt.datetime.today().strftime('%Y-%m-%d')

    df = yf.download(ticker, start=start, end=end)
    df1 = df.copy()

    df1.reset_index(drop=True,inplace=True)
    df1.drop(["Adj Close"], axis=1, inplace=True)
    
    return df1