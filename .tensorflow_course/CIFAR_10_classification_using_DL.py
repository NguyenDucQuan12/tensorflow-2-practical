import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import datetime
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 

print(tf.__version__)
import keras

# google colab: https://colab.research.google.com/drive/1C1fedenupkmeBB6e6yQijpLy7QIYpjPp

"""
CIFAR-10 is a dataset that consists of several images divided into the following 10 classes:

Airplanes
Cars
Birds
Cats
Deer
Dogs
Frogs
Horses
Ships
Trucks
The dataset stands for the Canadian Institute For Advanced Research (CIFAR)

CIFAR-10 is widely used for machine learning and computer vision applications.

The dataset consists of 60,000 32x32 color images and 6,000 images of each class.

Images have low resolution (32x32).

Data Source: https://www.cs.toronto.edu/~kriz/cifar.html
"""

# Tải dữ liệu từ bộ dữ liệu có sẵn của tensorflow
(X_train, y_train) , (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Hiển thị hình ảnh có thứ tự 300
i = 300
plt.imshow(X_train[i])
print(y_train[i])
plt.show()

# Hiển thị một tập dữ liệu trên ma trận 4x4 và kích thước của từng hình ảnh là 15x15
W_grid = 4
L_grid = 4

fig, axes = plt.subplots(L_grid, W_grid, figsize = (15, 15))
axes = axes.ravel()

n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    index = np.random.randint(0, n_training) # pick a random number
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')
    
plt.subplots_adjust(hspace = 0.4)
plt.show()

### Chuẩn bị và làm sạch dữ liệu
print(X_train)
# chuyển đổi thành bộ dữ liệu float 32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# số lượng các category
number_cat = 10

print(y_train)
# chuyển đổi bô dữ liệu y train thành định dạng ma trận 1 hàng 10 cột
y_train = keras.utils.to_categorical(y_train, num_classes= number_cat)

print(y_train)

y_test = keras.utils.to_categorical(y_test, num_classes= number_cat)
# chuẩn hóa dữ liệu huấn luyện (mục đích là đưa các giá trị của đầu vào nằm trong khoảng 0 đến 1)
X_train = X_train/255
X_test = X_test/255

print(X_train.shape)
Input_shape = X_train.shape[1:]
print(Input_shape)

### TẠO MODEL VÀ HUẤN LUYỆN NÓ
cnn = tf.keras.Sequential()

cnn.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
cnn.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'))
cnn.add(tf.keras.layers.MaxPooling2D(2,2))
cnn.add(tf.keras.layers.Dropout(0.3))


cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))
cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))
cnn.add(tf.keras.layers.MaxPooling2D(2,2))
cnn.add(tf.keras.layers.Dropout(0.3))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(1024, activation = 'relu'))
cnn.add(tf.keras.layers.Dropout(0.3))

cnn.add(tf.keras.layers.Dense(1024, activation = 'relu'))

cnn.add(tf.keras.layers.Dense(10, activation = 'softmax'))
cnn.summary()

# biên dịch mô hình, thử lại với hàm tối ưu Adam
cnn.compile(optimizer = tf.keras.optimizers.RMSprop(0.0001, decay = 1e-6), loss ='categorical_crossentropy', metrics =['accuracy'])

# huấn luyện mô hình với 100 epochs
epochs = 100

history = cnn.fit(X_train, y_train, batch_size = 512, epochs = epochs)

### ĐÁNH GIÁ MÔ HÌNH 
evaluation = cnn.evaluate(X_test, y_test)
print('Test Accuracy: {}'.format(evaluation[1]))

predict_x=cnn.predict(X_test) 
predicted_classes=np.argmax(predict_x,axis=1)

print(predicted_classes)

y_test = y_test.argmax(1)

L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)    
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(predicted_classes, y_test)
cm
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True)
plt.show()