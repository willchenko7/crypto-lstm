import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from train_test_split import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import matplotlib.dates as mdates


def get_previous_predicted_prices(symbol,data_frequency,indicators,model,scaler,n_steps,n_features,num_predictions=24):
    '''
    Goal: Return an array of the predicted price at each point in the last 300 hours
         It will also return the actual price at each point in the last 300 hours

    
    Inputs:
        symbol: string
            The symbol of the cryptocurrency to be analyzed
        data_frequency: string
            The frequency of the data to be analyzed
        indicators: list of strings
            The indicators to be used in the model
        model: tensorflow model
            The model to be used to make predictions
        scaler: sklearn scaler
            The scaler used to scale the data
        n_steps: int
            The number of steps to be used in the model
        n_features: int
            The number of features to be used in the model
        num_predictions: int
            The number of predictions to be made
            max=300-12-1=287
        
    Output:
        predicted_prices: numpy array
            An array of the predicted price at each point in the last 300 hours
        actual_prices: numpy array
            An array of the actual price at each point in the last 300 hours
    '''
    #load live data
    data = pd.read_csv(f'data/{symbol}_{data_frequency}_live_financial_indicators.csv')
    #save the time column
    times_df = data['time']
    data = data[indicators]
    #order this data  by date in descending order
    #data = data.sort_values(by='date', ascending=False)
    #get a subset of this data, get the latest 1000 rows
    #data = data.iloc[:1000]
    #times_df = times_df.iloc[:1000]
    #start at the 300th row, and get the previous n_steps rows as a subset of data
    #loop through the remainder of the data, so from 300 to the latest available row
    predicted_prices = []
    actual_prices = []
    times = []
    for i in range(num_predictions,0,-1):
        print(f'Predicting price for row {i}')
        #get the previous n_steps rows
        data_subset = data.iloc[i:i+n_steps]
        #scale the data
        data_subset = scaler.transform(data_subset)
        #reshape the data
        data_subset = data_subset.reshape(1, n_steps, n_features)
        #make a prediction
        prediction = model.predict(data_subset)
        #inverse transform the prediction
        prediction = scaler.inverse_transform(np.concatenate((prediction, np.zeros((len(prediction), n_features-1))), axis=1))[:, 0]
        #append the prediction to the predicted_prices array
        predicted_prices.append(prediction[0])
        #append the actual price to the actual_prices array
        actual_prices.append(data.iloc[i]['close'])
        times.append(times_df.iloc[i])
    return predicted_prices, actual_prices, times

def plot_actual_vs_predicted(predicted_prices,acutal_prices,times,model_name):
    x = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in times]
    plt.plot(x, predicted_prices, label='Predicted Price')
    plt.plot(x, actual_prices, label='Actual Price')
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=50)) # display x-axis tick every 2 hours
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    #rotate the x-axis tick labels by 45 degrees
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time (all times UTC)')
    plt.ylabel('Price')
    plt.title(f'Model: {model_name}')
    plt.legend()
    #plt.xticks(x, [date.strftime('%m-%d %H') for date in x], rotation=45, ha='right')
    #save fig
    plt.savefig(f'graphs/{model_name}_predicted_prices.png')
    return


if __name__ == '__main__':
    #Define inputs to get_previous_predicted_prices
    symbol = 'BTC-USD'
    indicators = ['close', 'fisher', 'macd', 'sma', 'wma']
    data_frequency = 'hour'
    #define hyperparameters
    n_steps = 30
    n_epochs = 500
    n_batch_size = 32
    n_features = len(indicators)
    train_percentage = 0.8
    #load scaler
    with open(f'data/{symbol}_{n_steps}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    #load saved model
    model_name = f'{symbol}_{n_steps}_{n_features}_{n_epochs}'
    model = tf.keras.models.load_model(f'models/{model_name}.h5')
    #get predicted prices
    predicted_prices,actual_prices,times = \
        get_previous_predicted_prices(symbol,data_frequency,indicators,model,scaler,n_steps,n_features)
    print(predicted_prices)
    print(actual_prices)
    print(times)
    #plot actual vs predicted
    plot_actual_vs_predicted(predicted_prices,actual_prices,times,model_name)