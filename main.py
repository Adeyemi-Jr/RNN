import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout




### Predicting google stock price in Jan 2017


##################################################
##################################################
            # Data preprocessing
##################################################
##################################################


# import training set
df_train  = pd.read_csv('../data/Google_Stock_Price_Train.csv')
training_set = df_train.iloc[:,1:2].values
#training_set = df_train['Open']


#Feature scaling - apply normalisation since we are using sigmoid
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#creating a data structure with 60 timesteps and 1 output

X_train = []
y_train = []

for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60 : i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

#reshape
X_train = np.reshape(X_train, ( X_train.shape[0], X_train.shape[1], 1 ))



##################################################
##################################################
            # Building RNN
##################################################
##################################################

#Initialise the RNN
regressor = Sequential()


#Add first LSTM layer and Dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True, input_shape= (X_train.shape[1], 1)) )
regressor.add(Dropout(0.2))

#Add 2nd LSTM layer and Dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True) )
regressor.add(Dropout(0.2))

#Add 3rd LSTM layer and Dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True) )
regressor.add(Dropout(0.2))

#Add 4rd LSTM layer and Dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True) )
regressor.add(Dropout(0.2))

#Add output layer
regressor.add(Dense(units = 1) ) #1 since were only predicting stock price at t+1

#Compile RNN
regressor.compile(optimizer= 'adam', loss= 'mean_squared_error' )

#fitting the RNN to the training set
regressor.fit( X_train, y_train, epochs=100, batch_size= 32)


##################################################
##################################################
            # Making Predictions
##################################################
##################################################

# import training set
df_test  = pd.read_csv('../data/Google_Stock_Price_Test.csv')
real_stock_price = df_test.iloc[:,1:2].values


#getting the predictied stock price of 2017
df_total = pd.concat((df_train['Open'], df_test['Open']), axis = 0 )
inputs = df_total[len(df_total)- len(df_test) - 60: ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60 : i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, ( X_test.shape[0], X_test.shape[1], 1 ))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualisation results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price Jan 2017')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price Jan 2017')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.grid()
plt.show()


A = 1

