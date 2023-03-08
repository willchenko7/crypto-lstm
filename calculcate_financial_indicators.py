import pandas as pd
import numpy as np

def calculate_financial_indicators(file_path):
    '''
    goal: calculate financial indicators such as macd, sma, wma, fisher, etc.

    input:
        file_path: path to the csv file containing the historical price data

    output:
        df containing the calculated financial indicators and historical price data

    '''
    #read the historical price data
    df = pd.read_csv(file_path)
    #normalize close price between -1 and 1, write this down as a new column
    df['norm_close'] = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
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
    #drop the rows with NaN values
    df = df.dropna()
    return df

if __name__ == '__main__':
    #calculate the financial indicators
    df = calculate_financial_indicators('data/BTC-USD_hour.csv')
    #save the dataframe as a csv file
    df.to_csv('data/BTC-USD_hour_financial_indicators.csv', index=False)