# data_download.py

import yfinance as yf

def download_data(ticker='RBLX', start_date='2010-01-01', end_date='2024-10-01'):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv('data/rblx.csv')

if __name__ == '__main__':
    download_data()