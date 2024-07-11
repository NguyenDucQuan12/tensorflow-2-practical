# Tối ưu hóa tensorflow với CPU
# https://colab.research.google.com/drive/1xwNiawxGiag04F5rDy_CZZZjbnnm63WX
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# print(tf.__version__)
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

"""

Dự đoán số lượng người thuê xe đạp công cộng dựa trên các dữ liệu đã có 
Data Description:

instant: record index
dteday : date
season : season (1:springer, 2:summer, 3:fall, 4:winter)
yr : year (0: 2011, 1:2012)
mnth : month ( 1 to 12)
hr : hour (0 to 23)
holiday : wether day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
weekday : day of the week
workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
weathersit :
1: Clear, Few clouds, Partly cloudy
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
temp : Normalized temperature in Celsius. The values are divided to 41 (max)
hum: Normalized humidity. The values are divided to 100 (max)
windspeed: Normalized wind speed. The values are divided to 67 (max)
casual: count of casual users
registered: count of registered users,
cnt: count of total rental bikes including both casual and registered
"""

temp_df = pd.read_csv(".tensorflow_course\\assests\\bike_sharing_daily.csv")
# print(temp_df.head())

### LÀM SẠCH DỮ LIỆU
# Kiểm tra giá trị null của dữ liệu
# sns.heatmap(temp_df.isnull())
# plt.show()

# Loại bỏ 1 cột, nó trùng với giá trị index, 2 cột khác không sử dụng
temp_df = temp_df.drop(labels=['instant'], axis=1)
temp_df = temp_df.drop(labels=['casual', 'registered'], axis=1)
# chuyển đổi thời gian của dataset từ 01/01/2001 thành  2001-01-01
temp_df.dteday = pd.to_datetime(temp_df.dteday, format="%m/%d/%Y")
# sử dụng thời gian để làm chỉ số index, thay vì số thứ tự 1,2,3,4,...
temp_df.index = pd.DatetimeIndex(temp_df.dteday)
# sau khi lấy thời gian làm index thì ta có thể loại bỏ cột thời gian
temp_df = temp_df.drop(labels=['dteday'], axis=1)
# print(temp_df) # Hiển thị lại dữ liệu để kiểm tra

### VẼ DATASET
# Vì đã chỉ định index là thời gian nên có thể vẽ biểu đồ theo tần suất theo từng tuần 'W'
temp_df['cnt'].asfreq('W').plot(linewidth = 3)
plt.title("Bike usage per week")
plt.xlabel("Week")
plt.ylabel("Bike Rental")
plt.show()

# Hiển thị tất cả biểu đồ của dataset
# sns.pairplot(temp_df)
# plt.show()

## Lấy ra các cột temp, hum, windspeed, cnt để xem sự tương quan giữa các dữ liệu
X_numerical = temp_df[['temp', 'hum', 'windspeed', 'cnt']]
sns.pairplot(X_numerical)
plt.show()
# Xem mối tương quan
sns.heatmap(X_numerical.corr(), annot=True)
plt.show()

### Tạo bộ dữ liệu training và test
X_category = temp_df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]
# chuyển đổi dữ liệu từ bảng thành ma trận
oneHotEncoder = OneHotEncoder()
X_category = oneHotEncoder.fit_transform(X= X_category).toarray()
# print(X_category)
# chuyển đổi nó thành dataframe
X_category = pd.DataFrame(X_category)

# Xóa chỉ số index của X_number để ghép vào X_category để làm dư liệu trainning
X_numerical = X_numerical.reset_index()
x_all = pd.concat([X_category, X_numerical], axis=1)
# loại bỏ cột dteday sau khi xóa bỏ chỉ mục
x_all = x_all.drop(['dteday'], axis=1)
# print(x_all)
# Chia tất cả dữ liệu thành x và y, x là input, y sẽ là output
x = x_all.iloc[:, :-1].values
y = x_all.iloc[:, -1 :].values # chỉ lấy cột cuối cùng làm giá trị y
# print(y)

scaler = MinMaxScaler()
# chuẩn hóa cột y về 0-1
y = scaler.fit_transform(y)
# print(y)
# chia bộ dữ liệu thành train và test sẽ là 8-2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

### Huấn luyện model
# Tạo model tuần tự với 3 lớp ẩn, 3 lớp ẩn gồm 100 node, hàm kích hoạt sử dụng là ReLU, đầu vào là 35 đặc điểm
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 100, activation = "relu", input_shape = (35,)))
model.add(tf.keras.layers.Dense(units = 100, activation = "relu"))
model.add(tf.keras.layers.Dense(units = 1, activation = "linear"))

# print(model.summary())
# Lựa chọn hàm tối ưu Adam và hàm mất mát là MSE
model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
# Huấn luyện model
epoch_hits = model.fit(x_train, y_train, epochs = 20, batch_size = 50, validation_split = 0.2)

# Đánh giá mô hình
# In các thông tin thu được sau mỗi lần huấn luyện
# Biểu đồ này hai đường phải xêm xêm nhau, không được chênh nhau quá xa, nếu như vậy thì dẫn đến có thể là overfit hoặc model chưa đủ tốt
print(epoch_hits.history.keys())
plt.plot(epoch_hits.history['loss'])
plt.plot(epoch_hits.history['val_loss'])
plt.title("Tranining and validation loss")
plt.show()

# Dự đoán với tập dữ liệu test
y_predict = model.predict(x_test)
plt.plot(y_test, y_predict, '^', color = 'r')
plt.show()

# Chuyển đổi các giá trị dự đoán về như định dạng ban đầu
y_predict_origin = scaler.inverse_transform(y_predict)
y_test_origin = scaler.inverse_transform(y_test)
plt.plot(y_predict_origin, y_test_origin, '^', color = 'r')
plt.show()

# Tính toán MSE
k = x_test.shape[1]
n = len(x_test)
RMSE= float(format(np.sqrt(mean_squared_error(y_test_origin, y_predict_origin)), '0.3f'))
print(RMSE)

MSE = mean_squared_error(y_test_origin, y_predict_origin)
MAE = mean_absolute_error(y_test_origin, y_predict_origin)
r2 = r2_score(y_test_origin, y_predict_origin)
adj_r2 = 1 - (1-r2)*(n-1)/(n-k-1)

print('MSE: ', MSE, '\nMAE: ', MAE, '\nr2: ', r2, '\nAdj_r2: ', adj_r2)