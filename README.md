# crypto-lstm

This repo contains the minimal set of building blocks for training and deploying a LSTM neural network that predicts the next price of a cryptocurrency. The code is writen to be easily modified to fit a user's particular requirements. As an example, this repo uses BTC-USD as the currency and frequeny of each price is 1 hr.

This repo contains python scripts for the following tasks:
* Obtaining historical crypto data, for training (get_hisorical_data.py)
* Obtaining live crypto data, for deploying (get_live_crypto_data.py)
* Calculating financial indicators, such as weighted moving average, macd, and the fisher transform (calculate_financial_indators.py)
* Splitting historical data into train sets and test set (train_test_spit.py)
* Creating and training an LSTM tensorflow model (nn_training.py)
* Tuning the hyperparameters of this model using grid search (hyperparameter-tuning.py)
* Plotting the performance of a given model (get_predicted_prices.py)
* Predicting the next live price (get_predicted_prices.py)

This repo contains the following folders for storing information:
* data - all live and historical data gets saved here. also sklearn scalers are saved here.
* models - contains all tensorflow models. General naming structure = {symbol}_{n_steps}_{n_epochs}_{n_batch_size}_{train_percentage}.h5
* graphs - contains all plots to compare predicted price vs actual, for various models