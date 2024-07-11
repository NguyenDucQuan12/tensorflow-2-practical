import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# print(tf.__version__)
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# https://colab.research.google.com/drive/1eiMl_DeuwX9kc05_YXfBMEf6elXCk5g_
"""
Dataset includes house sale prices for King County in USA.
Homes that are sold in the time period: May, 2014 and May, 2015.
Data Source: https://www.kaggle.com/harlfoxem/housesalesprediction
Columns:
ida: notation for a house
date: Date house was sold
price: Price is prediction target
bedrooms: Number of Bedrooms/House
bathrooms: Number of bathrooms/House
sqft_living: square footage of the home
sqft_lot: square footage of the lot
floors: Total floors (levels) in house
waterfront: House which has a view to a waterfront
view: Has been viewed
condition: How good the condition is ( Overall )
grade: overall grade given to the housing unit, based on King County grading system
sqft_abovesquare: footage of house apart from basement
sqft_basement: square footage of the basement
yr_built: Built Year
yr_renovated: Year when house was renovated
zipcode: zip
lat: Latitude coordinate
long: Longitude coordinate
sqft_living15: Living room area in 2015(implies-- some renovations)
sqft_lot15: lotSize area in 2015(implies-- some renovations)
"""

house_df = pd.read_csv(".tensorflow_course\\assests\\kc_house_data.csv")
# print(house_df.head(30))
# house_df.describe()

### HIỂN THỊ DATASET
sns.scatterplot(x = 'sqft_living', y = 'price', data = house_df)
plt.show()

house_df.hist(bins = 20, figsize = (20,20), color = 'g')
plt.show()

### TẠO BỘ DỮ LIỆU TEST VÀ TRAIN
# Tập trung vào các đặc điểm quan trọng
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']
X = house_df[selected_features]
# print(X)
Y = house_df['price']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = Y.values.reshape(-1,1)
Y_scaled = scaler.fit_transform(Y_scaled)

### HUẤN LUYỆN MODEL
# Tạo bộ dữ liệu huấn luyện và test
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.25)

# Tạo model với 3 lớp ẩn, mỗi lớp ẩn có 100 node và có hàm kích hoạt ReLU
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(7, )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
# model.summary()
model.compile(optimizer='Adam', loss='mean_squared_error')
# Huấn luyện model
epochs_hists = model.fit(X_train, Y_train, epochs = 100, batch_size = 50, validation_split = 0.2)


### ĐÁNH GIÁ MODEL
epochs_hists.history.keys()

plt.plot(epochs_hists.history['loss'])
plt.plot(epochs_hists.history['val_loss'])
plt.title("Tranining and validation loss")
plt.show()
# thử nghiệm với bộ dữ liệu test, mô phỏng dữ liệu nên sẽ là đường thẳng chéo góc 45 độ
y_predict = model.predict(X_test)
plt.plot(Y_test, y_predict, "^", color = 'r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')
plt.show()

# chuyển đổi giá trị về ban đầu và hiển thị
y_predict_orig = scaler.inverse_transform(y_predict)
y_test_orig = scaler.inverse_transform(Y_test)
plt.plot(y_test_orig, y_predict_orig, "^", color = 'r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')
plt.xlim(0, 5000000)
plt.ylim(0, 3000000)

# Tính toán các giá trị của model
k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 

### CẢI THIỆN MODEL BẰNG CÁCH THÊM CÁC ĐẶC ĐIỂM ĐỂ LÀM ĐẦU VÀO CHO MODEL (INPUT), VUI LÒNG COMMENT CÁC ĐOẠN CODE PHÍA TRÊN TRƯỚC KHI CHẠY CODE MỚI BÊN DƯỚI
selected_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors', 'sqft_above', 'sqft_basement', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'yr_built', 
'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

X = house_df[selected_features]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

y = house_df['price']
y = y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)

model = tf.keras.models.Sequential()
# Đã thêm 12 input nên đầu vào cảu model không còn alf 7 nữa mà là 19
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(19, )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.compile(optimizer='Adam', loss='mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, validation_split = 0.2)

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color = 'r')
plt.xlabel("Model Predictions")
plt.ylabel("True Value (ground Truth)")
plt.title('Linear Regression Predictions')
plt.show()

y_predict_orig = scaler.inverse_transform(y_predict)
y_test_orig = scaler.inverse_transform(y_test)

# Tính toán các giá trị của model
k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 