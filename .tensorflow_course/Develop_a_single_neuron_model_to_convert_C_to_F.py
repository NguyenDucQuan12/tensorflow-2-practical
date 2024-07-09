# Tối ưu hóa tensorflow với CPU
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# print(tf.__version__)
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


temp_df = pd.read_csv(".tensorflow_course\\assests\\dataset.csv")
# print(temp_df.tail(5))
# print(temp_df.describe())
# print(temp_df.info())

# # hiển thị data
# sns.scatterplot(data=temp_df, x='Celsius', y = 'Fahrenheit')
# plt.show()

# Chia dữ liệu và huấn luyện, do bộ dữ liệu ít nên dùng hết luôn, bình thường chia 8/2
x_train = temp_df['Celsius']
y_train = temp_df['Fahrenheit']
# print(x_train.shape, y_train.shape)

model = tf.keras.Sequential()
# Chỉ một neuron 
# model.add(tf.keras.layers.Dense(units = 1, input_shape = (1,)))

# 3 lớp ẩn, lớp thứ nhất và thứ 2 là 5 node(5 neuron), lớp thứ 3 chỉ có 1 đầu ra
model.add(tf.keras.layers.Dense(units = 5, input_shape = (1,)))
model.add(tf.keras.layers.Dense(units = 5, input_shape = (1,)))
model.add(tf.keras.layers.Dense(units = 1))

# print(model.summary())
model.compile(optimizer = tf.keras.optimizers.Adam(1.0), loss = 'mean_squared_error')
epochs_hits = model.fit(x_train, y_train, epochs = 100)

# Đánh giá mô hình
# Xem các giá trị sau khi huấn luyện mô hình và vẽ nó
# print(epochs_hits.history.keys()) # return loss
# print(epochs_hits.history['loss'])
# plt.plot(epochs_hits.history['loss'])
# plt.title("Loss during training")
# plt.xlabel("Epochs")
# plt.ylabel("Tranning loss")
# plt.legend(["Trainning loss"])
# plt.show()
# lấy giá trị weight
weight = model.get_weights()

# Dự đoán với 1 giá trị ngẫu nhiên
temp_C = 0
temp_F = model.predict(np.array([temp_C])) # Tensor chỉ chấp nhận giá trị numpy
print(temp_F)
