# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import torch
import torch.nn as nn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import os
import json
from os.path import join
import math
from sklearn.metrics import mean_squared_error

# Slice the data frame 
def split_data(stock, stock1, stock2, stock3, stock4, lookback):

    data_raw = stock.to_numpy() # convert to numpy array
    label_raw = stock1.to_numpy()

    hr_raw = stock2.to_numpy()
    sen_raw = np.array(stock3)
    vec_raw = np.array(stock4)

    hr_data = []
    price_data = []
    label = []
    ori_data = []
    sen_data = []
    vec_data = []
    
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
        
        # price data
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
        
    
    hr_data = np.array(hr_data)
    price_data = np.array(price_data)
    label = np.array(label)
    sen_data = np.array(sen_data)
    vec_data = np.array(vec_data)   
 
    ori_data = np.array(ori_data)
    return[price_data, hr_data, label, ori_data, sen_data, vec_data]

def createDataset(price_data, hr_data, label, ori_data, sen_data, vec_data): 
    
    test_set_size = int(np.round(0.2*hr_data.shape[0]))
    train_val_set_size = hr_data.shape[0] - (test_set_size)
    train_set_size = int(np.round(train_val_set_size * 0.8))
    
    print('total: ', hr_data.shape[0], 'train: ', train_set_size, 'validation: ', train_val_set_size-train_set_size, 'test: ', test_set_size)
    
    # Train Dataset
    x_train = price_data[:train_set_size,:-1,:]
    x_train1 = hr_data[:train_set_size,:-1,:] # Comment last line(line87) and change x_train1 in this line to x_train if you want to use "hashrate", do same process to validation and test part in following lines
    x_train2 = sen_data[:train_set_size,:-1]
    x_train2 = np.expand_dims(np.array(x_train2),2)
    x_train3 = vec_data[:train_set_size,:]
    x_train3 = np.expand_dims(np.array(x_train3),2)
    x_train = np.append(x_train, x_train1, axis = 1) # Comment this line if you want to use "price", or "hashrate", do same process to validation and test part in following lines
    x_train = np.append(x_train, x_train2, axis = 1) # Comment this line if you want to use "price", "price+hashrate", or "hashrate", do same process to validation and test part in following lines
    x_train = np.append(x_train, x_train3, axis = 1) # Comment this line if you want to use "price", "price+hashrate", or "hashrate", do same process to validation and test part in following lines
    
    y_train = label[:train_set_size,-1,:]
    
    # Label
    # the 8th day's original price
    label_train = ori_data[:train_set_size,-1,:]
    # the 8th day's scaled price
    label_train_scaled = price_data[:train_set_size,-1,:]
    # the 7th day's original price
    price_train = ori_data[:train_set_size,-2,:]
    # the 7th day's scaled price
    price_train_scaled = price_data[:train_set_size,-2,:]
    
    # Validation Dataset
    x_val = price_data[train_set_size:train_val_set_size,:-1,:]
    
    x_val1 = hr_data[train_set_size:train_val_set_size,:-1,:]
    x_val2 = sen_data[train_set_size:train_val_set_size,:-1]
    x_val2 = np.expand_dims(np.array(x_val2),2)
    x_val3 = vec_data[train_set_size:train_val_set_size,:]
    x_val3 = np.expand_dims(np.array(x_val3),2)
    x_val = np.append(x_val, x_val1, axis = 1)
    x_val = np.append(x_val, x_val2, axis = 1)  
    x_val = np.append(x_val, x_val3, axis = 1)
    
    y_val = label[train_set_size:train_val_set_size,-1,:]
    
    # Label
    # the 8th day's original price
    label_val = ori_data[train_set_size:train_val_set_size,-1,:]
    # the 8th day's scaled price
    label_val_scaled = price_data[train_set_size:train_val_set_size,-1,:]
    # the 7th day's original price
    price_val = ori_data[train_set_size:train_val_set_size,-2,:]
    # the 7th day's scaled price
    price_val_scaled = price_data[train_set_size:train_val_set_size,-2,:]    
    
    # Test 
    x_test = price_data[train_val_set_size:,:-1]
    
    x_test1 = hr_data[train_val_set_size:,:-1]
    x_test2 = sen_data[train_val_set_size:,:-1]
    x_test2 = np.expand_dims(np.array(x_test2),2)
    x_test3 = vec_data[train_val_set_size:,:]
    x_test3 = np.expand_dims(np.array(x_test3),2)
    x_test = np.append(x_test, x_test1, axis = 1)
    x_test = np.append(x_test, x_test2, axis = 1)  
    x_test = np.append(x_test, x_test3, axis = 1)
    
    y_test = label[train_val_set_size:,-1,:]
    
    # Lable
    # the 8th day's original data
    label_test = ori_data[train_val_set_size:,-1,:]
    # the 8th day's scaled data
    label_test_scaled = price_data[train_val_set_size:,-1,:]
    # the 7th day's original data
    price_test = ori_data[train_val_set_size:,-2,:]    
    # the 7th day's scaled data
    price_test_scaled = price_data[train_val_set_size:,-2,:]    
    
    return [x_train, y_train, label_train, price_train, label_train_scaled, price_train_scaled, x_val, y_val, label_val, price_val, label_val_scaled, price_val_scaled, x_test, y_test, label_test, price_test, label_test_scaled, price_test_scaled]

# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout = 0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
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

# biLSTM model
class biLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(biLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout = 0.2)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def save_model(s_dir, model_name, epoch, model, optimizer):
    
    if not os.path.isdir(s_dir):
        os.makedirs(s_dir)
    save_path = join(s_dir, model_name+'.pt')
    print("Saving checkpoint to {}".format(save_path))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, save_path) 

def Training(model, criterion, optimizer, num_epochs, mod_dir, save_dir, x_train, y_train_model, price_train_model, price_train_scaled_model, label_train_model, label_train_scaled_model, x_val, y_val_model, price_val_model, label_val_model, price_val_scaled_model, label_val_scaled_model):
        
    # Initialization
    validation_loss_list = []
    hist = np.zeros(num_epochs)
    num_x = []
    min_val_loss = 9999
    
    # Training
    print('Start Training ...')
    start_time = time.time()
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_model)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if((t+1)%600==0): # 600/250/
            optimizer.param_groups[0]['lr'] /= 1.5        
        
        # Validate loop
        with torch.no_grad():
            if ((t+1)%5 == 0):
                validation_loss = 0
                y_val_pred = model(x_val)
                validation_loss = criterion(y_val_pred, y_val_model).item() 
                num_x.append(t)
                validation_loss_list.append(validation_loss) 
                print('Validation set: {:d}, loss: {:.6f}'.format(len(validation_loss_list), validation_loss))  
                if (validation_loss < min_val_loss):
                    save_model(mod_dir, 'ckpt-{}-{:.8f}'.format(t, validation_loss), t, model, optimizer)
                    min_val_loss = validation_loss            
    
    # Calculating costing time
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))
    
    # Train result (price)
    predict = pd.DataFrame(y_train_pred.detach().numpy())
    pre_train = predict[0]
    new_train_prices = []
    new_scaled_train_prices = []
    
    for i in range(len(price_train_model)):
        new_train_prices.append(price_train_model[i] * (1 + pre_train[i])) 
        
    for i in range(len(price_train_scaled_model)):
        new_scaled_train_prices.append(price_train_scaled_model[i] * (1 + pre_train[i]))     
        
    # Validation result (price)
    predict_val = pd.DataFrame(y_val_pred.detach().numpy())
    pre_val = predict_val[0]
    new_val_prices = []
    new_scaled_val_prices = []
    for i in range(len(price_val_model)):
        new_val_prices.append(price_val_model[i] * (1 + pre_val[i]))     
    for i in range(len(price_val_scaled_model)):
        new_scaled_val_prices.append(price_val_scaled_model[i] * (1 + pre_val[i]))     
    # Display result
    print("================Train Result======================") 
    new_per = y_train_pred.detach().numpy().squeeze().tolist()
    ori_per = y_train_model.detach().numpy().squeeze().tolist()
    len_new_per = len(new_per)
    num_correct = 0
    num_correct1 = 0
    sign_correct = 0
    for i in range(len_new_per):
        if(new_per[i]*ori_per[i] > 0):
            sign_correct += 1
            if(abs(new_per[i]-ori_per[i])<0.05):
                num_correct += 1     
        if(abs(new_per[i]-ori_per[i])<0.05):
            num_correct1 += 1     
            
    acc1 = sign_correct/len_new_per
    acc21 = num_correct/len_new_per
    acc22 = num_correct1/len_new_per
    print('Predicted %:', new_per)
    print('Groundtruth %:', ori_per)
    print('Predicted price:', new_train_prices)# new 8th day's price
    print('Groundtruth price:', label_train_model)# old 8th day's price
    print('Predicted scaled price:', new_scaled_train_prices)# new 8th day's price
    print('Groundtruth scaled price:', label_train_scaled_model)# old 8th day's price    

    print('+/- correct: ', acc1)
    print('diff < 0.05: ', acc22)
    print('+/- correct & diff < 0.05: ', acc21) 
    
    # Print validation result
    print("================Validation Result======================") 
    new_per_val = y_val_pred.detach().numpy().squeeze().tolist()
    ori_per_val = y_val_model.detach().numpy().squeeze().tolist()
    len_new_per_val = len(new_per_val)
    num_correct_val = 0
    num_correct_val1 = 0
    sign_correct_val = 0
    for i in range(len_new_per_val):
        if(new_per_val[i]*ori_per_val[i] > 0):
            sign_correct_val += 1
            if(abs(new_per_val[i]-ori_per_val[i])<0.05):
                num_correct_val += 1    
        if(abs(new_per_val[i]-ori_per_val[i])<0.05):
            num_correct_val1 += 1        
    acc1_val = sign_correct_val/len_new_per_val
    acc21_val = num_correct_val/len_new_per_val
    acc22_val = num_correct_val1/len_new_per_val
    print('Predicted %:', new_per_val)
    print('Groundtruth %:', ori_per_val)
    print('Predicted price:', new_val_prices)# new 8th day's price
    print('Groundtruth price:', label_val_model)# old 8th day's price 
    print('Predicted scaled price:', new_scaled_val_prices)# new 8th day's price
    print('Groundtruth scaled price:', label_val_scaled_model)# old 8th day's price 
    
    print('+/- correct: ', acc1_val)
    print('diff < 0.05: ', acc22_val)  
    print('+/- correct & diff < 0.05: ', acc21_val)
    
    # Plot prediction
    
    fig2 = plt.figure(figsize = (15,12))
    fig2.canvas.set_window_title('BTC price (ori & prediction)')
    ax1 = fig2.add_subplot(2,1,1)
    ax1.set_title('BTC price (ori & training prediction)',fontsize=14, fontweight='bold')
    ax1.plot(price_train_gru, 'b', label="Original Price (USD)")
    ax1.plot(new_train_prices, 'r', label="Training Prediction (GRU)")
    ax1.set_ylabel("Price (USD)", size = 14)
    ax1.set_xlabel("Days", size = 14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    ax2 = fig2.add_subplot(2,1,2)
    ax2.set_title('BTC price (ori & validation prediction)',fontsize=14, fontweight='bold')
    ax2.plot(price_val_gru, 'b', label="Original Price (USD)")
    ax2.plot(new_val_prices, 'r', label="Validation Prediction (GRU)")
    ax2.set_ylabel("Price (USD)", size = 14)
    ax2.set_xlabel("Days", size = 14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)    
    plt.savefig(save_dir+'gru_btc_price_prediction.jpg')
    plt.show()
    
    fig21 = plt.figure(figsize = (15,12))
    fig21.canvas.set_window_title('BTC scaled price (ori & prediction)')
    ax1 = fig21.add_subplot(2,1,1)
    ax1.set_title('BTC price (scaled & training prediction)',fontsize=14, fontweight='bold')
    ax1.plot(price_train_scaled_gru, 'b', label="Scaled Price (USD)")
    ax1.plot(new_scaled_train_prices, 'r', label="Training Prediction (GRU)")
    ax1.set_ylabel("Price", size = 14)
    ax1.set_xlabel("Days", size = 14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    ax2 = fig21.add_subplot(2,1,2)
    ax2.set_title('BTC price (scaled & validation prediction)',fontsize=14, fontweight='bold')
    ax2.plot(price_val_scaled_gru, 'b', label="Scaled Price (USD)")
    ax2.plot(new_scaled_val_prices, 'r', label="Validation Prediction (GRU)")
    ax2.set_ylabel("Price", size = 14)
    ax2.set_xlabel("Days", size = 14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)    
    plt.savefig(save_dir+'gru_btc_price_prediction_scaled.jpg')
    plt.show()    
    
    # Plot loss line
    fig3 = plt.figure(figsize = (15,12))
    fig3.canvas.set_window_title('Training & Validating Loss (GRU)')
    ax1 = fig3.add_subplot(2,1,1)
    ax1.set_title('Training Loss (GRU)',fontsize=14, fontweight='bold')
    ax1.plot(hist, 'b',label = 'train loss')
    ax2 = fig3.add_subplot(2,1,2)
    ax2.set_title('Validating Loss (GRU)',fontsize=14, fontweight='bold')
    ax2.plot(num_x, validation_loss_list, 'r',label = 'validation loss')
    plt.xlabel("Epoch", size = 14)
    plt.ylabel("Loss", size = 14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.savefig(save_dir+'model_train&vali_loss.jpg')
    plt.show()
    
    return new_train_prices, new_val_prices, new_scaled_train_prices, new_scaled_val_prices, training_time

def Testing(model, x_test, y_test_model, label_test_model, price_test_model, label_test_scaled_model, price_test_scaled_model):
    
    print('Testing ...')
    y_test_pred = model(x_test)
    
    test_predict = pd.DataFrame(y_test_pred.detach().numpy())
    
    # Test result (price)
    pre_test = test_predict[0]
    new_test_prices = []
    new_scaled_test_prices = []
    for i in range(len(price_test_model)):
        new_test_prices.append(price_test_model[i] * (1 + pre_test[i]))
    for i in range(len(price_test_model)):
        new_scaled_test_prices.append(price_test_scaled_model[i] * (1 + pre_test[i]))   
        
        
    print("==================Test Result=====================") 
    new_per_test = y_test_pred.detach().numpy().squeeze().tolist()
    ori_per_test = y_test_model.detach().numpy().squeeze().tolist()
    
    len_new_per_test = len(new_per_test)
    num_x_test = range(1,(len_new_per_test+1))
    
    num_correct_test = 0
    num_correct_test1 = 0
    sign_correct_test = 0
    for i in range(len_new_per_test):
        if(new_per_test[i]*ori_per_test[i] > 0):
            sign_correct_test += 1
            if(abs(new_per_test[i]-ori_per_test[i])<0.05):
                num_correct_test += 1
        if(abs(new_per_test[i]-ori_per_test[i])<0.05):
            num_correct_test1 += 1                
    acc3 = sign_correct_test/len_new_per_test
    acc41 = num_correct_test/len_new_per_test
    acc42 = num_correct_test1/len_new_per_test
    
    print('Predicted %:', new_per_test)
    print('Groundtruth %:', ori_per_test)
    print('Predicted price:', new_test_prices)
    print('Groundtruth price:', label_test_model)  
    print('Predicted scaled price:', new_scaled_test_prices)
    print('Groundtruth scaled price:', label_test_scaled_model)     
    
    print('+/- correct: ', acc3)
    print('diff < 0.05: ', acc42)
    print('+/- correct & diff < 0.05: ', acc41)
    return new_test_prices, new_scaled_test_prices

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
    
def calculateScore_model(lookback, training_time, hr, ori_price, save_dir,label_train_model, new_train_prices, new_scaled_train_prices, label_train_scaled_model, label_test_model, new_test_prices, new_scaled_test_prices, label_test_scaled_model, new_val_prices, label_val_model, new_scaled_val_prices, label_val_scaled_model):    
    
    model_result = []
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(label_train_model, new_train_prices))
    print('Train Score: %.2f RMSE' % (trainScore))
    trainScore_scaled = math.sqrt(mean_squared_error(label_train_scaled_model, new_scaled_train_prices))
    print('Train Score (scaled data): %.2f RMSE' % (trainScore_scaled))    
    
    valScore = math.sqrt(mean_squared_error(label_val_model, new_val_prices))
    print('Validation Score: %.2f RMSE' % (valScore)) 
    valScore_scaled = math.sqrt(mean_squared_error(label_val_scaled_model, new_scaled_val_prices))
    print('Validation Score (scaled data): %.2f RMSE' % (valScore_scaled))     
    
    testScore = math.sqrt(mean_squared_error(label_test_model, new_test_prices))
    print('Test Score: %.2f RMSE' % (testScore))
    testScore_scaled = math.sqrt(mean_squared_error(label_test_scaled_model, new_scaled_test_prices))
    print('Test Score (scaled data): %.2f RMSE' % (testScore_scaled))    
    
    model_result.append(trainScore)
    model_result.append(trainScore_scaled)
    model_result.append(valScore)
    model_result.append(valScore_scaled)
    model_result.append(testScore)
    model_result.append(testScore_scaled)    
    model_result.append(training_time)
    
    show_train_prices = np.expand_dims(np.array(new_train_prices),1)
    show_val_prices = np.expand_dims(np.array(new_val_prices),1)
    show_test_prices = np.expand_dims(np.array(new_test_prices),1)
    
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(hr)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(new_train_prices)+lookback, :] = show_train_prices
    
    # shift validation predictions for plotting
    valPredictPlot = np.empty_like(hr)
    valPredictPlot[:, :] = np.nan
    valPredictPlot[len(new_train_prices)+lookback-1:len(new_train_prices)+len(new_val_prices)+lookback-1, :] = show_val_prices    
    
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(hr)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(new_train_prices)+len(new_val_prices)+lookback-1:len(hr)-1, :] = show_test_prices
    
    #original = scaler.inverse_transform(price['Closing Price (USD)'].values.reshape(-1,1))
    
    predictions = np.append(trainPredictPlot, valPredictPlot, axis=1)
    predictions = np.append(predictions, testPredictPlot, axis=1)
    predictions = np.append(predictions, ori_price, axis=1)
    result = pd.DataFrame(predictions)
    
    fig3 = plt.figure(figsize = (15,9))
    fig3.canvas.set_window_title('Results (GRU)')
    plt.plot(result.index, result[0], 'r', label="Train prediction")
    plt.plot(result.index, result[1], 'b', label="Validation prediction")
    plt.plot(result.index, result[2], 'g', label="Test prediction")
    plt.plot(result.index, result[3], 'y', label="Actual Value", alpha=0.8)
    plt.title('Results (GRU)',fontsize=14, fontweight='bold')
    plt.xlabel("Days", size = 14)
    plt.ylabel("Cost (USD)", size = 14)
    plt.legend(loc='best')
    plt.savefig(save_dir+'Results(GRU).jpg')
    plt.show()
    
    return model_result

def main():
    mo_use = 'gru'
    patten = 'all'
    
    save_dir = './result/result/'+ patten +'/'+ mo_use + '/'
    mod_dir = './result/model/'+ patten +'/'+ mo_use + '/'
    
    tweet_filepath = './dataset/tweet_2014-2021.json'
    filepath = './dataset/BTC_2014-2021.csv'
    data = pd.read_csv(filepath)
    data = data.sort_values('Date')
    data.head()
    
    # Print original data 
    fig1 = plt.figure(figsize = (15,9))
    fig1.canvas.set_window_title('Bitcoin Price')
    ax1 = fig1.add_subplot(111)
    ax1.plot(data[['Hash Rate']], 'r', label = 'hr')
    ax1.set_title("Blockchain Hash rate",fontsize=18, fontweight='bold')
    ax1.set_ylabel('Hashrate',fontsize=18)
    ax1.legend(loc='best')
    ax2 = ax1.twinx()
    ax2.plot(data[['Percentage Change']], 'b', label = 'Increasing Rate')
    ax1.set_ylabel('Percentage',fontsize=18)
    #ax2.set_xticks(range(0,data.shape[0],500),data['Date'].loc[::500],rotation=45)
    ax2.set_xlabel('Date',fontsize=18)
    ax2.legend(loc='best')
    plt.savefig(save_dir+'hash_rate.jpg')
    plt.show()
    
    # Load Data
    
    #price = data[['Closing Price (USD)']]
    ori_price = data[['Closing Price (USD)']]
    
    hr = data[['Hash Rate']]
    per = data[['Percentage Change']]
    
    # Tweet
    tweet_date = [] 
    tweet_sen = [] 
    tweet_vec = []  
    with open (tweet_filepath) as openfileobject:
        tweet_data = json.load(openfileobject)
        index = 0
        for item in tweet_data:
            tweet_date.append(item['date']) 
            tweet_sen.append(item['Sentiment_Analysis'])
            tweet_vec.append(item['key_word_table'])    
    
    
    # Create Dataset
    lookback = 8  # choose sequence length  
    price_data, hr_data, label, ori_data, sen, vec = split_data(ori_price, per, hr, tweet_sen, tweet_vec, lookback)
    
    
    x_train, y_train, label_train, price_train, label_train_scaled, price_train_scaled, x_val, y_val, label_val, price_val, label_val_scaled, price_val_scaled, x_test, y_test, label_test, price_test, label_test_scaled, price_test_scaled = createDataset(price_data, hr_data, label, ori_data,sen, vec)
    
    
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_val.shape = ', x_val.shape)
    print('y_val.shape = ', y_val.shape)    
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)
    print('price_train.shape = ', price_train.shape)
    print('label_train.shape = ', label_train.shape)
    print('price_train_scaled.shape = ', price_train_scaled.shape)
    print('label_train_scaled.shape = ', label_train_scaled.shape)    
    print('price_val.shape = ', price_val.shape)
    print('label_val.shape = ', label_val.shape)  
    print('price_val_scaled.shape = ', price_val_scaled.shape)
    print('label_val_scaled.shape = ', label_val_scaled.shape)     
    print('price_test.shape = ', price_test.shape)
    print('label_test.shape = ', label_test.shape)
    print('price_test_scaled.shape = ', price_test_scaled.shape)
    print('label_test_scaled.shape = ', label_test_scaled.shape)     

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    x_val = torch.from_numpy(x_val).type(torch.Tensor)
    y_train_model = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_model = torch.from_numpy(y_test).type(torch.Tensor)
    y_val_model = torch.from_numpy(y_val).type(torch.Tensor)
    price_train_model = price_train.squeeze()
    label_train_model = label_train.squeeze().tolist()
    price_train_scaled_model = price_train_scaled.squeeze()
    label_train_scaled_model = label_train_scaled.squeeze().tolist()    
    price_val_model = price_val.squeeze()
    label_val_model = label_val.squeeze().tolist()
    price_val_scaled_model = price_val_scaled.squeeze()
    label_val_scaled_model = label_val_scaled.squeeze().tolist()    
    price_test_model = price_test.squeeze()
    label_test_model = label_test.squeeze().tolist()
    price_test_scaled_model = price_test_scaled.squeeze()
    label_test_scaled_model = label_test_scaled.squeeze().tolist()    
    
    # Model Parameters Initialization
    input_dim = 1
    hidden_dim = 36 # GRU_PRICE:17; LSTM_PRICE:32;
    num_layers = 2
    output_dim = 1
    num_epochs = 1500
    runTrain = True
    runTest = True
    
    # Initialize Model
    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    if(runTrain):
        # Train
        new_train_prices, new_val_prices, new_scaled_train_prices, new_scaled_val_prices, training_time = Training(model, criterion, optimizer, num_epochs, mod_dir, save_dir, 
                                                                                                                  x_train, y_train_model, price_train_model, price_train_scaled_model, label_train_model, label_train_scaled_model, 
                                                                                                                  x_val, y_val_model, price_val_model, label_val_model, price_val_scaled_model, label_val_scaled_model)
    if(runTest):
        # Load Saved Model
        checkpoint_path = get_checkpoint(mod_dir)
        checkpoint = torch.load(checkpoint_path)
        loaded_model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)    
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        # Test
        new_test_prices, new_scaled_test_prices = Testing(loaded_model, x_test, y_test_model, label_test_model, price_test_model, label_test_scaled_model, price_test_scaled_model)   
    
    if(runTrain&runTest):
        # Calculating score for train or test
        model_result = calculateScore_model(lookback, training_time, hr, ori_price, save_dir, 
                                   label_train_model, new_train_prices, new_scaled_train_prices, label_train_scaled_model,
                                   label_test_model, new_test_prices, new_scaled_test_prices, label_test_scaled_model, 
                                   new_val_prices, label_val_model, new_scaled_val_prices, label_val_scaled_model)
    
        model_result = pd.DataFrame(model_result, columns=['GRU'])
        result = pd.concat([model_result], axis=1, join='inner')
        result.index = ['Train RMSE', 'Train RMSE (scaled data)', 'Validation RMSE', 'Validation RMSE (scaled data)', 'Test RMSE', 'Test RMSE (scaled data)', 'Train Time']
        print(result)
    
main()