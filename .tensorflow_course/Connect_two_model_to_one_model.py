# google_colab: https://colab.research.google.com/drive/1Q6_P5NSDBOM46VWJA_EU_f1bRBX7UAGx
# dataset: https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes
import tensorflow as tf 
import tensorflow_probability as tfp  # pip install tensorflow-probability==0.17.0  mới tương thích với tf 2.10
import cv2   # pip install opencv-python
import numpy as np
import seaborn as sns # pip install seaborn
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.regularizers  import L2, L1
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, roc_curve # pip install scikit-learn

### DATASETS
train_direct = "Python_310/.tensorflow_gpu/assets/Human_emotions_datasets/Emotions Dataset/train"
val_direct = "Python_310/.tensorflow_gpu/assets/Human_emotions_datasets/Emotions Dataset/test"
CLASS_NAME = ["angry","happy","sad"]

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

# chuyển đổi nhãn của dataset từ int sang dạng one-hot endcoding
# 0: tức giận 1: hạnh phúc 2: buồn chuyển thành (1 0 0): tức giận, (0 1 0): hạnh phúc, (0 0 1): buồn
def preprocess(image, label):
    label = tf.one_hot(label, depth=len(CLASS_NAME))
    return image, label
# Lấy bộ dữ liệu từ thư mục đã phân loại sẵn, chỉ có tác dụng trên tensorflow verdion 2.10 trở về (phiên bản hỗ trợ GPU)
train_datasets = tf.keras.utils.image_dataset_from_directory(
    train_direct,
    labels='inferred',  # tự động suy ra nhãn từ tên thư mục
    label_mode='int',  # 'int' cho nhãn số, 0: tức giận, 1: hạnh phúc, 2: buồn
    class_names=CLASS_NAME,# Tên class 
    color_mode='rgb',
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    seed=99,
    shuffle=True,
).map(preprocess)

val_datasets = tf.keras.utils.image_dataset_from_directory(
    val_direct,
    labels='inferred',
    label_mode='int',
    class_names=CLASS_NAME,
    color_mode='rgb',
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
).map(preprocess)

augment_layers = tf.keras.Sequential([
  RandomRotation(factor = (-0.025, 0.025)), #Giới hạn góc xoay chỉ để mặt người hơi nghiêng : -0.025*360 = 9 độ, giới hạn quay từ -9 đến 9
  RandomFlip(mode='horizontal',),
  RandomContrast(factor=0.1),
])

@tf.function
def augment_layer(image, label):
   return augment_layers(image, training = True), label

## Cắt và trộn 2 hình ảnh vào nhau bằng hàm box và cutmix
def box(lamda):

  r_x = tf.cast(tfp.distributions.Uniform(0, CONFIGURATION["IM_SIZE"]).sample(1)[0], dtype = tf.int32)
  r_y = tf.cast(tfp.distributions.Uniform(0, CONFIGURATION["IM_SIZE"]).sample(1)[0], dtype = tf.int32)

  r_w = tf.cast(CONFIGURATION["IM_SIZE"]*tf.math.sqrt(1-lamda), dtype = tf.int32)
  r_h = tf.cast(CONFIGURATION["IM_SIZE"]*tf.math.sqrt(1-lamda), dtype = tf.int32)

  r_x = tf.clip_by_value(r_x - r_w//2, 0, CONFIGURATION["IM_SIZE"])
  r_y = tf.clip_by_value(r_y - r_h//2, 0, CONFIGURATION["IM_SIZE"])

  x_b_r = tf.clip_by_value(r_x + r_w//2, 0, CONFIGURATION["IM_SIZE"])
  y_b_r = tf.clip_by_value(r_y + r_h//2, 0, CONFIGURATION["IM_SIZE"])

  r_w = x_b_r - r_x
  if(r_w == 0):
    r_w  = 1

  r_h = y_b_r - r_y
  if(r_h == 0):
    r_h = 1

  return r_y, r_x, r_h, r_w

def cutmix(train_dataset_1, train_dataset_2):
  (image_1,label_1), (image_2, label_2) = train_dataset_1, train_dataset_2

  lamda = tfp.distributions.Beta(2,2)
  lamda = lamda.sample(1)[0]

  r_y, r_x, r_h, r_w = box(lamda)
  crop_2 = tf.image.crop_to_bounding_box(image_2, r_y, r_x, r_h, r_w)
  pad_2 = tf.image.pad_to_bounding_box(crop_2, r_y, r_x, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"])

  crop_1 = tf.image.crop_to_bounding_box(image_1, r_y, r_x, r_h, r_w)
  pad_1 = tf.image.pad_to_bounding_box(crop_1, r_y, r_x, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"])

  image = image_1 - pad_1 + pad_2

  lamda = tf.cast(1- (r_w*r_h)/(CONFIGURATION["IM_SIZE"]*CONFIGURATION["IM_SIZE"]), dtype = tf.float32)
  label = lamda*tf.cast(label_1, dtype = tf.float32) + (1-lamda)*tf.cast(label_2, dtype = tf.float32)

  return image, label

train_dataset_1 = train_datasets.map(augment_layer, num_parallel_calls = tf.data.AUTOTUNE)
train_dataset_2 = train_datasets.map(augment_layer, num_parallel_calls = tf.data.AUTOTUNE)

mixed_dataset = tf.data.Dataset.zip((train_dataset_1, train_dataset_2))

training_dataset = (
    mixed_dataset
    .map(cutmix, num_parallel_calls = tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = (
    val_datasets
    .prefetch(tf.data.AUTOTUNE)
)

### HẾT TĂNG CƯỜNG DỮ LIỆU


### TẠO MODEL 1
# Chuẩn hóa đầu vào
resize_rescale_layers = tf.keras.Sequential([
       Resizing(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
       Rescaling(1./255),
])

# Khởi tạo model có sẵn từ Tensorflow: https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB4
## Khởi tạo model có sẵn từ tensorflow để học chuyển giao (EfficientNetB5)
backbone = tf.keras.applications.efficientnet.EfficientNetB5(
    include_top = False,
    weights='imagenet',
    input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3),
    )

backbone.trainable=False


# backbone.trainable = True # Cho phép cập nhật các tham số của mô hình cũ, tuy nhiên lúc này phải đặt learning rate cực nhỏ (lr/100 hoặc lr/1000)s
backbone.trainable = False # Đóng băng các tham số của mô hình không cho huấn luyện lại

### TẠO MODEL BẰNG CÁCH HỌC CHUYỂN GIAO TỪ VGG16
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Dense(CONFIGURATION["N_DENSE_1"], activation = "relu")(x)
x = Dense(CONFIGURATION["N_DENSE_2"], activation = "relu")(x)
out_put = Dense(CONFIGURATION["NUM_CLASSES"], activation = "softmax")(x)

pretrain_model = Model(backbone.inputs, out_put) # vgg_backbone.inputs ko phải vgg_backbone.input
pretrain_model.summary()

### HUẤN LUYỆN

loss_function = CategoricalCrossentropy()

metrics = [CategoricalAccuracy(name = "accuracy"), TopKCategoricalAccuracy(k = 2, name = "top_k_accuracy")]

pretrain_model.compile(optimizer= Adam(learning_rate = CONFIGURATION["LEARNING_RATE"]),
                                loss = loss_function,
                                metrics = metrics,
                                ) # run_eagerly=True thêm tham số này khi trong huấn luyện có lỗi

history = pretrain_model.fit(train_datasets, 
                          validation_data= validation_dataset,
                          epochs= CONFIGURATION["N_EPOCHS"],
                          verbose=1)


### KIỂM TRA HIỆU SUẤT CỦA MODEL
pretrain_model.evaluate(validation_dataset)

### LƯU MODEL
# Một cách lưu và tải model thường được sử dụng ở tensorflow phiên bản 2.10 trở về, còn mới hơn thì phải thêm đuôi ".keras"
pretrain_model.save("Python_310/.tensorflow_gpu/assets/save_model/Human_emotions_detect_pretrain_EfficientNetB5")

### TẢI MODEL THỨ 2 HOẶC HUẤN LUYỆN MODEL KHÁC
vgg_model_transfer = tf.keras.models.load_model("Python_310/.tensorflow_gpu/assets/save_model/Human_emotions_detect_pretrain_vgg16")
vgg_model_transfer.summary()


### KẾT NỐI 2 MODEL VỚI NHAU
inputs = Input(shape = (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3))
y_1 = vgg_model_transfer(inputs)
y_2 = pretrain_model(inputs)

out_put = 0.5*y_1 + 0.5*y_2
ensemble_model = Model(inputs = inputs, outputs = out_put)

ensemble_model.compile(optimizer= Adam(learning_rate = CONFIGURATION["LEARNING_RATE"]),
                                loss = loss_function,
                                metrics = metrics,
                                ) # run_eagerly=True thêm tham số này khi trong huấn luyện có lỗi

### KIỂM TRA HIỆU SUẤT CỦA MODEL
ensemble_model.evaluate(validation_dataset)