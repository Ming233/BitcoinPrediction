import torch
import torch.nn as nn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import os
from os.path import join
import math
import json
from sklearn.metrics import mean_squared_error

def split_data(stock, stock1, stock2, stock3, stock4, stock5, lookback):

    data_raw = stock.to_numpy() # convert to numpy array
    label_raw = stock1.to_numpy()
    hr_raw = stock2.to_numpy()
    date_raw = stock3.to_numpy()
    sen_raw = np.array(stock4)
    vec_raw = np.array(stock5)    
    
    hr_data = []
    label = []
    ori_data = []
    sen_data = []
    vec_data = []
    price_data = []
    date = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        
        # original data
        ori_data.append(data_raw[index: index + lookback])
        
        # hash rate 
        # scaled data
        scaler = MinMaxScaler(feature_range=(0, 1))
        d = hr_raw[index: index + lookback]
        d = scaler.fit_transform(d.reshape(-1,1))
        hr_data.append(d)
        
        # price 
        # scaled data
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        p = data_raw[index: index + lookback]
        p = scaler1.fit_transform(p.reshape(-1,1))
        price_data.append(p)        
        
        # label
        label.append(label_raw[index: index + lookback])  
        
        # tweet data
        sen_data.append(sen_raw[index: index + lookback])
        
        vec_array_raw = vec_raw[index: index + lookback-1]
        
        vec_array = vec_array_raw.sum(axis=0) 
        vec_data.append(vec_array)
        #print(vec_data)
        
        # date
        date.append(date_raw[index: index + lookback])        
        
    
    hr_data = np.array(hr_data)
    price_data = np.array(price_data)
    label = np.array(label)
    ori_data = np.array(ori_data)
    sen_data = np.array(sen_data)
    vec_data = np.array(vec_data)  
    date = np.array(date)
    
    #print("========================")
    #print(sen_data.shape)
    #print(vec_data.shape)
    #print(hr_data.shape)
    #print(label.shape)
    #print(ori_data.shape)
    #print("========================")
    return[price_data, hr_data, label, ori_data, sen_data, vec_data, date]


def createDataset(price_data, hr_data, label, date, ori_data, sen_data, vec_data, ): 
    
    # Test 
    x_test = price_data[:,:-1]
    x_test1 = hr_data[:,:-1]
    x_test2 = sen_data[:,:-1]
    x_test2 = np.expand_dims(np.array(x_test2),2)
    x_test3 = vec_data[:,:]
    x_test3 = np.expand_dims(np.array(x_test3),2)
    x_test = np.append(x_test, x_test1, axis = 1)
    x_test = np.append(x_test, x_test2, axis = 1)  
    x_test = np.append(x_test, x_test3, axis = 1) 
    
    y_test = label[:,-1,:]
    
    ori_x_test = ori_data[:,:-1]
    # Lable
    # the 8th day's original data
    label_test = ori_data[:,-1,:]
    # the 7th day's data
    price_test = ori_data[:,-2,:]
    
    test_date = date[:,-1,:]
    
    return [x_test, y_test, ori_x_test, label_test, price_test, test_date]

# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    
# biLSTM model
class biLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(biLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    
# GRU model
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def Testing(model, x_test, y_test_lstm, label_test_lstm, price_test_lstm):
    print('Testing ...')
    y_test_pred = model(x_test)
    
    test_predict = pd.DataFrame(y_test_pred.detach().numpy())
    
    # Test result (price)
    pre_test = test_predict[0]
    #ori_test = test_original[0]
    new_test_prices = []
    
    for i in range(len(price_test_lstm)):
        new_test_prices.append(price_test_lstm[i] * (1 + pre_test[i]))
        
    print("==================Test Result=====================") 
    new_per_test = y_test_pred.detach().numpy().squeeze().tolist()
    ori_per_test = y_test_lstm.detach().numpy().squeeze().tolist()
    
    len_new_per_test = len(new_per_test)
    num_x_test = range(1,(len_new_per_test+1))
    
    #print('new %:', new_per_test)
    #print('old %:', ori_per_test)    
    
    num_correct_test = 0
    sign_correct_test = 0
    for i in range(len_new_per_test):
        if(new_per_test[i]*ori_per_test[i] > 0):
            sign_correct_test += 1
            if(abs(new_per_test[i]-ori_per_test[i])<0.05):
                num_correct_test += 1
                
    acc3 = sign_correct_test/len_new_per_test
    acc4 = num_correct_test/len_new_per_test

    print('+/- correct: ', acc3)
    print('diff < 0.05: ', acc4)
    #print('new price:', new_test_prices)
    #print('old price:', label_test_lstm)
    return new_test_prices

# Load model
def get_checkpoint(ckpt_dir):
    """return full path if there is ckpt in ckpt_dir else None"""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('ckpt')]
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))
    last_ckpt, max_epoch = None, 0
    for ckpt in ckpts:
        epoch = int(ckpt.split('-')[1])
        if epoch >= max_epoch:
            max_epoch = epoch
            last_ckpt = ckpt
    full_path = os.path.join(ckpt_dir, last_ckpt)
    print("Get checkpoint from {} for training".format(full_path))
    return full_path 


save_dir = './result_shuffle/result/OutputPrice/'
tweet_filepath = './dataset/tweet_2014-2021.json'
filepath = './dataset/BTC_2014-2021.csv'
data = pd.read_csv(filepath)
data = data.sort_values('Date')
data.head()

# Load Data
#price = data[['Closing Price (USD)']]
ori_price = data[['Closing Price (USD)']]
hr = data[['Hash Rate']]
per = data[['Percentage Change']]
date = data[['Date']]

tweet_date = [] 
tweet_sen = [] 
tweet_vec = []  
with open (tweet_filepath) as openfileobject:
    tweet_data = json.load(openfileobject)
    index = 0
    #tweet_data.reverse()
    for item in tweet_data:
        tweet_date.append(item['date']) 
        tweet_sen.append(item['Sentiment_Analysis'])
        tweet_vec.append(item['key_word_table']) 

# Create Dataset
lookback = 8  # choose sequence length

price_data, hr_data, label, ori_data, sen_data, vec_data, testdate = split_data(ori_price, per, hr, date, tweet_sen, tweet_vec, lookback)
x_test, y_test, ori_x_test, label_test, price_test, test_date = createDataset(price_data, hr_data, label, testdate, ori_data, sen_data, vec_data)

test_date = test_date.tolist()
price_test = price_test.squeeze()
label_test = label_test.squeeze().tolist()
y_test = torch.from_numpy(y_test).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)

# Model Parameters Initialization
input_dim = 1

num_layers = 2
output_dim = 1


# Choose the feature combination you want
patten = 'all'

# Load Saved Model
# For LSTM
hidden_dim = 64 # price: 32; hr: 32; hr+price: 64; all: 64
checkpoint_path_lstm = './result_shuffle/model/bestFORtest/lstm_all_ckpt-84-0.00182903.pt'# get_checkpoint(mod_dir+'lstm/')
checkpoint_lstm = torch.load(checkpoint_path_lstm)
loaded_model_lstm = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)    
loaded_model_lstm.load_state_dict(checkpoint_lstm['model_state_dict'])
# Test
new_test_prices_lstm = Testing(loaded_model_lstm, x_test, y_test, label_test, price_test) 

# For bilstm
hidden_dim = 64 # price: 32; hr: 17; hr+price: 32; all: 64
checkpoint_path_bilstm = './bestFORtest/bilstm_all_ckpt-84-0.00183515.pt'
checkpoint_bilstm = torch.load(checkpoint_path_bilstm)
loaded_model_bilstm = biLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)    
loaded_model_bilstm.load_state_dict(checkpoint_bilstm['model_state_dict'])
# Test
new_test_prices_bilstm = Testing(loaded_model_bilstm, x_test, y_test, label_test, price_test) 

# For GRU
hidden_dim = 36 # price: 17; hrL 32; hr+price: 17; all: 36
checkpoint_path_gru = './bestFORtest/gru_all_ckpt-99-0.00183528.pt'
checkpoint_gru = torch.load(checkpoint_path_gru)
loaded_model_gru = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)    
loaded_model_gru.load_state_dict(checkpoint_gru['model_state_dict'])
# Test
new_test_prices_gru = Testing(loaded_model_gru, x_test, y_test, label_test, price_test) 


def saveJson(data,patten,save_dir):
    print('Save file!')
    name = save_dir + 'Output_'+patten + '.json'
    with open(name, 'w') as outfile:
        json.dump(data, outfile) 

saveOutput = []
for i in range(len(new_test_prices_gru)):
    d = {}
    d['correct price'] = str(label_test[i])
    d['gru_predicted_price'] = str(new_test_prices_gru[i])
    d['lstm_predicted_price'] = str(new_test_prices_lstm[i])
    d['bilstm_predicted_price'] = str(new_test_prices_bilstm[i])
    d['Date'] = test_date[i]
    saveOutput.append(d) 
print(len(saveOutput))
print(saveOutput)
saveJson(saveOutput, patten, save_dir)  

