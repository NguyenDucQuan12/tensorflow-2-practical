import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# print(tf.__version__)
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import layers
import PIL

# Feature Scaling is a must in ANN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Google colab: https://colab.research.google.com/drive/1ITPkPyXdfdCp4TWIcF9JfsurUZHT6Rdy#scrollTo=-uEFeTF3rPl0

"""

In this case study, you have been provided with images of traffic signs and the goal is to train a Deep Network to classify them
The dataset contains 43 different classes of images.
Classes are as listed below:
( 0, b'Speed limit (20km/h)') ( 1, b'Speed limit (30km/h)')
( 2, b'Speed limit (50km/h)') ( 3, b'Speed limit (60km/h)')
(4, b'Speed limit (70km/h)') ( 5, b'Speed limit (80km/h)')
( 6, b'End of speed limit (80km/h)') ( 7, b'Speed limit (100km/h)')
( 8, b'Speed limit (120km/h)') ( 9, b'No passing')
(10, b'No passing for vehicles over 3.5 metric tons')
(11, b'Right-of-way at the next intersection') (12, b'Priority road')
(13, b'Yield') (14, b'Stop') (15, b'No vehicles')
(16, b'Vehicles over 3.5 metric tons prohibited') (17, b'No entry')
(18, b'General caution') (19, b'Dangerous curve to the left')
(20, b'Dangerous curve to the right') (21, b'Double curve')
(22, b'Bumpy road') (23, b'Slippery road')
(24, b'Road narrows on the right') (25, b'Road work')
(26, b'Traffic signals') (27, b'Pedestrians') (28, b'Children crossing')
(29, b'Bicycles crossing') (30, b'Beware of ice/snow')
(31, b'Wild animals crossing')
(32, b'End of all speed and passing limits') (33, b'Turn right ahead')
(34, b'Turn left ahead') (35, b'Ahead only') (36, b'Go straight or right')
(37, b'Go straight or left') (38, b'Keep right') (39, b'Keep left')
(40, b'Roundabout mandatory') (41, b'End of no passing')
(42, b'End of no passing by vehicles over 3.5 metric tons')
"""

with open(".tensorflow_course\\assests\\traffic-signs-data\\train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open(".tensorflow_course\\assests\\traffic-signs-data\\valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open(".tensorflow_course\\assests\\traffic-signs-data\\test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)

X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print(X_train.shape)

### HIỂN THỊ MỘT SỐ DATA
i = 3000
plt.imshow(X_train[i])
print(y_train[i])
plt.show()

### CHUẨN BỊ DỮ LIỆU 
# Chuyển đổi hình ảnh màu thành hỉnh ảnh xám( chỉ có 1 chiều)
X_train, y_train = shuffle(X_train, y_train)

X_train_gray = np.sum(X_train/3, axis = 3, keepdims = True)
X_test_gray = np.sum(X_test/3, axis = 3, keepdims = True)
X_validation_gray = np.sum(X_validation/3, axis = 3, keepdims = True)
print(X_train_gray.shape)

# chuẩn hóa dataset
X_train_gray_norm = (X_train_gray - 128)/128
X_test_gray_norm = (X_test_gray - 128)/128
X_validation_gray_norm = (X_validation_gray - 128)/128

print(X_train_gray_norm)

# hiển thị hình ảnh lại
i = 60
plt.imshow(X_train_gray[i].squeeze(), cmap = 'gray')
plt.figure()
plt.imshow(X_train[i])
plt.figure()
plt.imshow(X_train_gray_norm[i].squeeze(), cmap = 'gray')
plt.show()

i = 500
plt.imshow(X_validation_gray[i].squeeze(), cmap = 'gray')
plt.figure()
plt.imshow(X_validation[i])
plt.figure()
plt.imshow(X_validation_gray_norm[i].squeeze(), cmap = 'gray')
plt.show()

### HUẤN LUYỆN MODEL
"""
STEP 1: THE FIRST CONVOLUTIONAL LAYER #1
Input = 32x32x1
Output = 28x28x6
Output = (Input-filter+1)/Stride* => (32-5+1)/1=28
Used a 5x5 Filter with input depth of 3 and output depth of 6
Apply a RELU Activation function to the output
pooling for input, Input = 28x28x6 and Output = 14x14x6
* Stride is the amount by which the kernel is shifted when the kernel is passed over the image.
STEP 2: THE SECOND CONVOLUTIONAL LAYER #2

Input = 14x14x6
Output = 10x10x16
Layer 2: Convolutional layer with Output = 10x10x16
Output = (Input-filter+1)/strides => 10 = 14-5+1/1
Apply a RELU Activation function to the output
Pooling with Input = 10x10x16 and Output = 5x5x16
STEP 3: FLATTENING THE NETWORK

Flatten the network with Input = 5x5x16 and Output = 400
STEP 4: FULLY CONNECTED LAYER

Layer 3: Fully Connected layer with Input = 400 and Output = 120
Apply a RELU Activation function to the output
STEP 5: ANOTHER FULLY CONNECTED LAYER

Layer 4: Fully Connected Layer with Input = 120 and Output = 84
Apply a RELU Activation function to the output
STEP 6: FULLY CONNECTED LAYER

Layer 5: Fully Connected layer with Input = 84 and Output = 43

"""

from tensorflow.keras import datasets, layers, models

LeNet = models.Sequential()

LeNet.add(layers.Conv2D(6, (5,5), activation = 'relu', input_shape = (32,32,1)))
LeNet.add(layers.AveragePooling2D(2,2))


LeNet.add(layers.Conv2D(16, (5,5), activation = 'relu'))
LeNet.add(layers.AveragePooling2D(2,2))

LeNet.add(layers.Flatten())

LeNet.add(layers.Dense(120, activation = 'relu'))

LeNet.add(layers.Dense(84, activation = 'relu'))

LeNet.add(layers.Dense(43, activation = 'softmax'))
LeNet.summary()

LeNet.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = LeNet.fit(X_train_gray_norm,
                 y_train,
                 batch_size = 500,
                 epochs = 50,
                 verbose = 1,
                 validation_data = (X_validation_gray_norm, y_validation))

### ĐÁNH GIÁ MÔ HÌNH
score = LeNet.evaluate(X_test_gray_norm, y_test)
print('Test Accuracy: {}'.format(score[1]))

history.history.keys()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

# dự đoán với bộ thử nghiệm
# predicted_classes = LeNet.predict_classes(X_test_gray_norm) #đã lỗi thời, ko còn dùng được nữa

predict_x=LeNet.predict(X_test_gray_norm) 
predicted_classes=np.argmax(predict_x,axis=1)
y_true = y_test

# ma trận nhầm lẫn 
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)
plt.show()

# Hiển thị một số hình ảnh dự đoán và thực tế
L = 7
W = 7

fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_true[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)
plt.show()