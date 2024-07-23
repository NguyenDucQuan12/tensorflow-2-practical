import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf### models
import pandas as pd ### reading and processing data
import seaborn as sns ### visualization
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
from tensorflow.keras.layers import Dense, InputLayer

# google colab: https://colab.research.google.com/drive/1xJ49_6Z6AfUgeGmcsGbj1BaM48PbgIVy

second_hand_car_pd = pd.read_csv(".tensorflow_course/assests/second_hand_car_dataset.csv", sep= ",")
# print(second_hand_car_pd)

# sns.pairplot(second_hand_car_pd[['years', 'km', 'rating', 'condition', 'economy', 'top speed', 'hp', 'torque', 'current price']], diag_kind='kde')
# plt.show()
tensor_data = tf.constant(second_hand_car_pd)
# chuyển đổi kiểu dữ liệu thành float 32
tensor_data = tf.cast(tensor_data, tf.float32)

# Phân chia dữ liệu đầu vào và đầu ra
# Input chỉ lấy data từ cột thứ 3 đến cột gần cuối
X = tensor_data[:,3:-1]
# out put sẽ là cột cuối cùng
y = tensor_data[:,-1]
### CHIA BỘ DỮ LIỆU THÀNH BỘ TEST VÀ TRAIN LÀ 80,20
print(X.shape)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(X)

X_train = X[:int(DATASET_SIZE*TRAIN_RATIO)]
y_train = y[:int(DATASET_SIZE*TRAIN_RATIO)]
print(X_train.shape)
print(y_train.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

# bộ dữ liệu xác thực
X_val = X[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
y_val = y[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
print(X_val.shape)
print(y_val.shape)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

X_test = X[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
y_test = y[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
print(X_test.shape)
print(y_test.shape)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
                             InputLayer(input_shape = (8,)),
                             Dense(28, activation = "relu"),
                             Dense(28, activation = "relu"),
                             Dense(64, activation = "relu"),
                             Dense(1, activation = "sigmoid"),
])
model.summary()

model.compile(optimizer = "adam",
              loss = "binary_crossentropy",
               metrics =['accuracy'])

history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 100, verbose = 1)
