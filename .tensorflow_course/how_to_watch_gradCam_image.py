# google_colab: https://colab.research.google.com/drive/1Q6_P5NSDBOM46VWJA_EU_f1bRBX7UAGx
# dataset: https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes
import tensorflow as tf 
import tensorflow_probability as tfp  # pip install tensorflow-probability==0.17.0  mới tương thích với tf 2.10
import cv2   # pip install opencv-python
import numpy as np
import seaborn as sns # pip install seaborn
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input)
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense)


CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
    "LEARNING_RATE": 0.001,
    "N_EPOCHS": 20,
    "DROPOUT_RATE": 0.3,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 24,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 128,
    "N_DENSE_3": 64,
    "NUM_CLASSES": 3,
    "PATCH_SIZE": 16,
    "PROJ_DIM": 768,
    "CLASS_NAMES": ["angry", "happy", "sad"],
}

## Khởi tạo model có sẵn từ tensorflow để học chuyển giao (EfficientNetB5)
backbone = tf.keras.applications.efficientnet.EfficientNetB5(
    include_top = False,
    weights='imagenet',
    input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3),
    )
backbone.trainable=False

x = backbone.output

x = GlobalAveragePooling2D()(x)
x = Dense( CONFIGURATION["N_DENSE_1"], activation = "relu")(x)
x = Dense( CONFIGURATION["N_DENSE_2"], activation = "relu")(x)
output = Dense( CONFIGURATION["NUM_CLASSES"], activation = "softmax")(x)

#Tạo model để bắt đầu học chuyển giao
pretrained_model = Model(backbone.inputs, output)
pretrained_model.summary()
# Tải weight sau khi đào tạo hoặc tải model sau khi đã đào tạo, lưu ý model phải có lớp activation thì mới xem gradcam được
# ví dụ model EfficeientNetB5 sau khi chạy lệnh summary thì sẽ có lớp activation cuối cùng, sau khi đào tạo xong thì ta sẽ lấy lớp đó ở bên dưới
pretrained_model.load_weights("links")

img_path = "Python_310/.tensorflow_gpu/assets/Human_emotions_datasets/Emotions Dataset/test/happy/993500.jpg_rotation_1.jpg"

test_image = cv2.imread(img_path)
test_image = cv2.resize(test_image, (CONFIGURATION["IM_SIZE"] ,CONFIGURATION["IM_SIZE"]))
im = tf.constant(test_image, dtype = tf.float32)
img_array = tf.expand_dims(im, axis = 0)
print(img_array.shape)

preds = pretrained_model.predict(img_array)
print(preds)

### GET TOP ACTIVATION LAYER
last_conv_layer_name = "top_activation" # lấy hàm Activation cuối cùng, model.summary để xem hàm Activation cuối cùng có tên là gì
last_conv_layer = pretrained_model.get_layer(last_conv_layer_name)
last_conv_layer_model = Model(pretrained_model.input, last_conv_layer.output)

classifier_layer_names = [
 "global_average_pooling2d",
 "dense",
 "dense_1",
 "dense_2"
]

classifier_input = Input(shape=(8,8,2048))
x = classifier_input
for layer_name in classifier_layer_names:
    x = pretrained_model.get_layer(layer_name)(x)
classifier_model = Model(classifier_input, x)

with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    print(top_pred_index)
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, last_conv_layer_output)

print(grads.shape)

pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
print(pooled_grads.shape)


last_conv_layer_output = last_conv_layer_output.numpy()[0]
for i in range(2048):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]
print(last_conv_layer_output.shape)

heatmap = np.sum(last_conv_layer_output, axis=-1)

heatmap=tf.nn.relu(heatmap)
plt.matshow(heatmap)
plt.show()

resized_heatmap=cv2.resize(np.array(heatmap),(256,256))
plt.matshow(resized_heatmap*255+img_array[0,:,:,0]/255)
plt.show()