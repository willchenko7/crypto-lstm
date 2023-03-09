import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle


def train_test_split(data, train_percentage=0.8,n_steps=30):
    '''
    Goal: Generic function to split a pandas dataframe 
        into training and testing sets

    Input:
        data: pandas df
        train_percentage: what percentage of the data should be used for training
            default: 0.8
        n_steps: number of time steps to use for the LSTM model
            default: 30
        
    Output:
        train_data: pandas df containing the training data
        test_data: pandas df containing the testing data
        X_train: numpy array containing the training data
        y_train: numpy array containing the training labels
        X_test: numpy array containing the testing data
        y_test: numpy array containing the testing labels
        scaler: sklearn MinMaxScaler object used to scale the data, used for inverse scaling later
    '''
    #remove nan, inf, -inf from data
    data =data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    # Create the training data
    X_train = []
    y_train = []
    for i in range(n_steps, len(train_scaled)):
        X_train.append(train_scaled[i-n_steps:i])
        y_train.append(train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Create the testing data
    X_test = []
    y_test = []
    for i in range(n_steps, len(test_scaled)):
        X_test.append(test_scaled[i-n_steps:i])
        y_test.append(test_scaled[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    return train_data, test_data, X_train, y_train, X_test, y_test, scaler

if __name__ == '__main__':
    symbol = 'BTC-USD'
    frequency = 'hour'
    n_steps = 30
    #load historical data
    data = pd.read_csv(f'data/{symbol}_{frequency}_financial_indicators.csv')
    data = data[['close', 'fisher', 'macd', 'sma', 'wma']]
    #train test split
    train_data, test_data, X_train, y_train, X_test, y_test, scaler = \
        train_test_split(data,train_percentage=0.8, n_steps=n_steps)
    #save the scaler as a pickle file
    with open(f'data/{symbol}_{n_steps}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)