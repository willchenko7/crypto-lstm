import pandas as pd
import numpy as np

def calculate_financial_indicators(file_path,live=False):
    '''
    goal: calculate financial indicators such as macd, sma, wma, fisher, etc.

    input:
        file_path: path to the csv file containing the historical price data

    output:
        df containing the calculated financial indicators and historical price data

    '''
    #read the historical price data
    df = pd.read_csv(file_path)
    #order the dataframe by date in ascending order
    df = df.sort_values('time')
    #only get first 1000 rows of data
    df = df.iloc[:1000]
    #if this is live data, we need to get the min and max of the close price of the historical data
    if live:
        #read the historical price data
        historical_df = pd.read_csv(f'{file_path[:-4]}_financial_indicators.csv')
        #get the min and max of the close price of the historical data
        min_close = historical_df['close'].min()
        max_close = historical_df['close'].max()
    else:
        #get the min and max of the close price of current df
        min_close = df['close'].min()
        max_close = df['close'].max()
    #normalize close price between -1 and 1, write this down as a new column
    #df['norm_close'] = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
    df['norm_close'] = (df['close'] - min_close) / (max_close - min_close)
    #calculate the fisher transform indicator
    df['fisher'] = 0.5 * np.log((1 + df['norm_close']) / (1 - df['norm_close']))
    df['fisher'] = df['fisher'].shift(1)
    #calculate the macd indicator
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    #calculate the simple moving average indicator
    df['sma'] = df['close'].rolling(window=10).mean()
    #calculate the weighted moving average indicator
    weights = np.arange(1, 11)
    weights = weights/weights.sum()
    df['wma'] = df['close'].rolling(window=10).apply(lambda x: (weights * x).sum(), raw=True)
    #order the dataframe by date in descending order
    df = df.sort_values('time', ascending=False)
    #drop the rows with NaN values
    df = df.dropna()
    return df

if __name__ == '__main__':
    #define inputs
    symbol = 'BTC-USD'
    frequency = 'hour'
    #calculate the financial indicators
    df = calculate_financial_indicators(f'data/{symbol}_{frequency}.csv', live=True)
    #save the dataframe as a csv file
    df.to_csv(f'data/{symbol}_{frequency}_financial_indicators-1000.csv', index=False)