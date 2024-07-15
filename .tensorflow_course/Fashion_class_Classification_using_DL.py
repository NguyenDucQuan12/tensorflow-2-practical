import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 

from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

### VISUAL CONVOLUTION NEURON NETWORK 2D
# https://setosa.io/ev/image-kernels/

"""
Fashion training set consists of 70,000 images divided into 60,000 training and 10,000 testing samples. Dataset sample consists of 28x28 grayscale image, associated with a label from 10 classes.

The 10 classes are as follows:

0 => T-shirt/top
1 => Trouser
2 => Pullover
3 => Dress
4 => Coat
5 => Sandal
6 => Shirt
7 => Sneaker
8 => Bag
9 => Ankle boot
Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255

"""

# print(tf.__version__)
### ĐỌC DỮ LIỆU TỪ CSV
fashion_train_df = pd.read_csv('.tensorflow_course\\assests\\fashion-mnist_train.csv',sep=',')
fashion_test_df = pd.read_csv('.tensorflow_course\\assests\\fashion-mnist_test.csv', sep = ',')

# Có thể sử dụng cùng 1 bộ dư liệu như trên bằng keras như bên dưới, định dạng dữ liệu sẵn là numpy.array
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# print(fashion_train_df)
# print(fashion_train_df.shape)

# chuyển đổi dừ liệu từ dataframe thành np.array để huân luyện model
training = np.array(fashion_train_df, dtype='float32')
testing = np.array(fashion_train_df, dtype='float32')

# Hiển thị một hình ảnh ngẫu nhiên và label của nó 
i = random.randint(1,60000)
plt.imshow(training[i,1:].reshape(28,28), cmap= 'gray')
plt.show()
label = training[i, 0]
print(label)

# Hiển thị nhiều hình ảnh và label của nó bên dưới, ma trận (15x15)
# Define the dimensions of the plot grid 
W_grid = 15
L_grid = 15

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_training = len(training) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index    
    axes[i].imshow( training[index,1:].reshape((28,28)) )
    axes[i].set_title(training[index,0], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
plt.show()

### XÂY DỰNG VÀ HUẤN LUYỆN MODEL
# Lấy tất cả dữ liệu của training bắt đầu từ cột số 2, cột đầu tiên là label nên sẽ ko lấy
x_train = training[:,1:]/255
y_train = training[:,0]

X_test = testing[:,1:]/255
y_test = testing[:,0]
# X_train.shape = (60000, 784) không thể làm đầu vào cho mô hình được, phải chuyển về (60000,28,28,1)

x_train = x_train.reshape(x_train.shape[0], *(28,28,1))
X_test = X_test.reshape(X_test.shape[0], *(28,28,1))
# Tạo model
cnn = models.Sequential()
cnn.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (28,28,1)))
cnn.add(layers.MaxPooling2D(2,2))
cnn.add(layers.Conv2D(64, (3,3), activation='relu'))
cnn.add(layers.MaxPooling2D(2,2))

cnn.add(layers.Conv2D(64, (3,3), activation='relu'))

cnn.add(layers.Flatten())

cnn.add(layers.Dense(64, activation='relu'))
cnn.add(layers.Dense(10, activation='softmax'))

cnn.summary()


cnn.compile(loss = 'sparse_categorical_crossentropy', optimizer= 'Adam', metrics=['accuracy'])
epochs = 150 # nên để 100 với máy tính cá nhân 
history = cnn.fit(x_train, y_train, batch_size= 512, epochs= epochs)

### ĐÁNH GIÁ MÔ HÌNH
# Mô hình có thể đã overfit, bởi vì độ chính xác gần như là 100%
evaluation = cnn.evaluate(X_test, y_test)
# Độ chính xác với tập thử nghiệm là 9x%, tuy nó khá cao nhwung tỉ lệ mô hình overfit cũng rất cao, vì vậy chúng ta
# sẽ thwucj hiện lại việc huấn luyện mô hình bằng cách chia bộ dữ liệu huấn luyện có thêm một phần xác thwucj nữa, validation

predicted_classes = cnn.predict_classes(X_test)
# Hiển thị các dự đoán cùng nhãn của nó lên biểu đồ
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # 

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)
plt.show()

cm = confusion_matrix(predicted_classes, y_test)
plt.figure(figsize= (14,10))
sns.heatmap(cm, annot=True)
plt.show()

num_class = 10
target_name = ['class{}'.format(i) for i in range(num_class)]
print(classification_report(y_test, predicted_classes, target_names= target_name))
