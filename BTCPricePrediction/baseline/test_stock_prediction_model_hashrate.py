# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import torch
import torch.nn as nn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import time
import os
import math
from sklearn.metrics import mean_squared_error
#import plotly
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py
import chart_studio

save_dir = './result/bit_hr1/'

filepath = './dataset/BTC_2014-2021.csv'
data = pd.read_csv(filepath)
data = data.sort_values('Date')
data.head()

#sns.set_style("darkgrid")
fig1 = plt.figure(figsize = (15,9))
fig1.canvas.set_window_title('Blockchain Hash rate')
plt.plot(data[['Hash Rate']])
plt.xticks(range(0,data.shape[0],500),data['Date'].loc[::500],rotation=45)
plt.title("Blockchain Hash rate",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Hash rate',fontsize=18)
plt.legend(loc='best')
#plt.show()
plt.savefig(save_dir+'Data_hash_rate.jpg')

fig11 = plt.figure(figsize = (15,9))
fig11.canvas.set_window_title('BTC Price')
plt.plot(data[['Closing Price (USD)']])
plt.xticks(range(0,data.shape[0],500),data['Date'].loc[::500],rotation=45)
plt.title("BTC Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('BTC Price',fontsize=18)
plt.legend(loc='best')
#plt.show()
plt.savefig(save_dir+'Data_btc_price.jpg')



hr = data[['Hash Rate']]
price = data[['Closing Price (USD)']]

hr.info()
price.info()

# Slice the data frame 
scaler = MinMaxScaler(feature_range=(-1, 1))
hr['Hash Rate'] = scaler.fit_transform(hr['Hash Rate'].values.reshape(-1,1))
#scaler = MinMaxScaler(feature_range=(-1, 1))
price['Closing Price (USD)'] = scaler.fit_transform(price['Closing Price (USD)'].values.reshape(-1,1))

def split_data(stock, stock1, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    label_raw = stock1.to_numpy()
    data = []
    label = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
        label.append(label_raw[index: index + lookback])
    
    data = np.array(data)
    label = np.array(label)
    
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = label[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = label[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

lookback = 7 # 20 # choose sequence length

x_train, y_train, x_test, y_test = split_data(hr, price, lookback)

print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

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

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))
predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

#sns.set_style("darkgrid")    

fig2 = plt.figure(figsize = (15,12))
fig2.canvas.set_window_title('BTC price & Training Loss (LSTM)')
ax1 = fig2.add_subplot(2,1,1)
plt.plot(original.index, original[0], 'b', label="Data")
plt.plot(predict.index, predict[0], 'r', label="Training Prediction (LSTM)")
plt.title('BTC price',fontsize=14, fontweight='bold')
plt.xlabel("Days", size = 14)
plt.ylabel("Cost (USD)", size = 14)
plt.legend(loc='best')
ax2 = fig2.add_subplot(2,1,2)
plt.plot(hist, 'b')
plt.title("Training Loss", size = 14, fontweight='bold')
plt.xlabel("Epoch", size = 14)
plt.ylabel("Loss", size = 14)
plt.legend(loc='best')
#plt.show()
plt.savefig(save_dir+'lstm_btc_price&train_loss.jpg')

# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
lstm.append(trainScore)
lstm.append(testScore)
lstm.append(training_time)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

# shift test predictions for plotting
testPredictPlot = np.empty_like(price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

original = scaler.inverse_transform(price['Closing Price (USD)'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)

fig3 = plt.figure(figsize = (15,9))
fig3.canvas.set_window_title('Results (LSTM)')
plt.plot(result.index, result[0], 'b', label="Train prediction")
plt.plot(result.index, result[1], 'r', label="Test prediction")
plt.plot(result.index, result[2], 'g', label="Actual Value")
plt.title('Results (LSTM)',fontsize=14, fontweight='bold')
plt.xlabel("Days", size = 14)
plt.ylabel("Cost (USD)", size = 14)
plt.legend(loc='best')
#plt.show()
plt.savefig(save_dir+'Results(LSTM).jpg')

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
    
model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(num_epochs)
start_time = time.time()
gru = []

for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train_gru)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time()-start_time    
print("Training time: {}".format(training_time))
predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))

#sns.set_style("darkgrid")    

fig4 = plt.figure(figsize = (15,12))
fig4.canvas.set_window_title('BTC price & Training Loss (GRU)')
ax1 = fig4.add_subplot(2,1,1)
plt.plot(original.index, original[0], 'b', label="Data")
plt.plot(predict.index, predict[0], 'r', label="Training Prediction (GRU)")
plt.title('BTC price',fontsize=14, fontweight='bold')
plt.xlabel("Days", size = 14)
plt.ylabel("BTC price", size = 14)
plt.legend(loc='best')
ax2 = fig4.add_subplot(2,1,2)
plt.plot(hist, 'b')
plt.title("Training Loss", size = 14, fontweight='bold')
plt.xlabel("Epoch", size = 14)
plt.ylabel("Loss", size = 14)
#plt.legend(loc='best')
#plt.show()
plt.savefig(save_dir+'gru_btc_price&train_loss.jpg')

# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_gru.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
gru.append(trainScore)
gru.append(testScore)
gru.append(training_time)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

# shift test predictions for plotting
testPredictPlot = np.empty_like(price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

original = scaler.inverse_transform(price['Closing Price (USD)'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)

fig5 = plt.figure(figsize = (15,9))
fig5.canvas.set_window_title('Results (GRU)')
plt.plot(result.index, result[0], 'b', label="Train prediction")
plt.plot(result.index, result[1], 'r', label="Test prediction")
plt.plot(result.index, result[2], 'g', label="Actual Value")
plt.title('Results (GRU)',fontsize=14, fontweight='bold')
plt.xlabel("Days", size = 14)
plt.ylabel("Cost (USD)", size = 14)
plt.legend(loc='best')
#plt.show()
plt.savefig(save_dir+'Results(GRU).jpg')

lstm = pd.DataFrame(lstm, columns=['LSTM'])
gru = pd.DataFrame(gru, columns=['GRU'])
result = pd.concat([lstm, gru], axis=1, join='inner')
result.index = ['Train RMSE', 'Test RMSE', 'Train Time']
print(result)