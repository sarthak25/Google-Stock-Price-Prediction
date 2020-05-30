# A PREDICTIVE MODEL FOR GOOGLE STOCK PRICES USING LINEAR REGRESSION
# BY - OMKAR VIVEK SABNIS: 17-06-2018

# IMPORTING ALL THE MODULES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# READING THE DATASET AND VISUALIZING A FEW ENTRIES
dt=pd.read_csv("Dataset/Google_Stock_Price_Train.csv")
print("Few Entries of the dataset:")
print(dt.head())

# PREPARATION OF THE TRAINING DATASET
df=dt.iloc[:,1:2].values # NUMPY ARRAY
print("Shape of the training dataset:")
print(dt.shape)

x_train=[]
y_train=[]
for i in range(60,1258):
    x_train.append(df[i-60:i,0])
    y_train.append(df[i,0])
x_train,y_train= np.array(x_train), np.array(y_train)
print("Converted X_train shape:")
print(x_train.shape)
print("Converted Y_train shape:")
print(y_train.shape)

# PREPARATION OF THE TESTING DATASET
df_test = pd.read_csv('Dataset/Google_Stock_Price_Test.csv')
Real_stock_price = df_test.iloc[:,1:2].values
print("Shape of the testing dataset:")
print(Real_stock_price.shape)

total_dataset = pd.concat((dt['Open'], df_test['Open']),axis = 0 )
inputs = total_dataset[len(total_dataset)-len(df_test) - 60:].values

inputs = inputs.reshape(-1,1)
print("Shape of the dataset after conversion:")
print(inputs.shape)

x_test = []
for i in range (60,80):
    x_test.append(inputs[i-60:i,0])

x_test=np.array(x_test)
print("Converted X_test shape:")
print(x_test.shape)

# FITTING THE DATA IN THE MODEL
lrm = LinearRegression()
print(lrm.fit(x_train,y_train))
pred_lrm = lrm.predict(x_test)
print("Prediction:")
print(pred_lrm)

# PLOTTING THE GRAPH
plt.plot(Real_stock_price,color='red', label='Real')
plt.plot(pred_lrm,color='green',label='Predicted')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

