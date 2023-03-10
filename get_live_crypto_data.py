import pandas as pd
import requests
import time
import os
from calculcate_financial_indicators import calculate_financial_indicators


def get_live_crypto_data(symbol,frequency):
    '''
    Goal Get the 300 latest data points for a given cryptocurrency 

    Inputs:
        symbol (str): The symbol of the cryptocurrency to get data for (e.g., BTC-USD).
        frequency (str): The frequency of the data (e.g., 'day', 'hour', 'minute').
    Returns:
        df (pandas dataframe): The dataframe containing the data
    '''
    # Define the API endpoint URL
    endpoint = f"https://api.pro.coinbase.com/products/{symbol}/candles"
    if frequency == "day":
        s_granularity = "86400"
    elif frequency == "hour":
        s_granularity = "3600"
    elif frequency == "minute":
        s_granularity = "60"
    else:
        print("Error: Invalid frequency")
        return None
    # Set the parameters for the API request
    params = {
        "granularity": s_granularity,
        "start": "",
        "end": "",
    }
    response = requests.get(endpoint, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
    else:
        # If the request failed, print an error message and return None
        print(f"Error: API request failed with status code {response.status_code}")
        print(response.json())
        return None
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

if __name__ == "__main__":
    #define inputs
    symbol = "BTC-USD"
    frequency = "hour"
    df = get_live_crypto_data(symbol,frequency)
    filename = f"data/{symbol}_{frequency}_live.csv"
    #save to csv
    df.to_csv(filename, index=False)
    #calculate financial indicators
    financial_df = calculate_financial_indicators(filename,live=True)
    #save to csv
    filename = f"data/{symbol}_{frequency}_live_financial_indicators.csv"
    financial_df.to_csv(filename, index=False)
    