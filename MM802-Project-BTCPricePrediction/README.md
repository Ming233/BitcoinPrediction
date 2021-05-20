# Bitcoin prediction machine-learning model experiment

## Required package
Before you run this code you need install all requirement package first:

### [Pytorch](https://pytorch.org/get-started/locally/)
`pip3 install torch`

### [scikit-learn](https://scikit-learn.org/stable/install.html)
`pip install -U scikit-learn`

### [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
`pip install pandas`

## User Guide
### baseline
This folder include the baseline for our prediction model refer from [Rodolfo Saldanha's article](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632)

#### dataset and result
These folders save the dataset used to test the following two code and their results.
[Bitcoin history price download](https://www.coindesk.com/price/bitcoin)
[Blockchain history Hash rate download](https://www.quandl.com/data/BCHAIN/HRATE-Bitcoin-Hash-Rate)

#### stock_prediction_pytorch.py
This is Rodolfo Saldanha's original code

#### test_stock_prediction_model_BTCprice.py
This is using BTC price to test on source code

#### test_stock_prediction_model_BTCprice.py
This is using hash rate to test on source code

### bitcoin_code
#### dataset, result, and result_shuffle
These folders save the dataset used in the following two code and their results.

#### predictBTC_all.py
experiment LSTM, bi-LSTM, GRU inputing four feature combinations.
you can change the model name in line 530 and the pattern value in line 531.

#### predictBTC_all_shuffle.py
experiment LSTM, bi-LSTM, GRU inputing four feature combinations.
you can change the model name in line 461 and the pattern value in line 462.

#### calculatePrice.py
This code is used to calculate the predicted price using the output (predicted increasing rate of price).
