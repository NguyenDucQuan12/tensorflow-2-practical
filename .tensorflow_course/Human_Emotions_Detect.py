# google_colab: https://colab.research.google.com/drive/1Q6_P5NSDBOM46VWJA_EU_f1bRBX7UAGx
# dataset: https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes
import tensorflow as tf 
import tensorflow_probability as tfp  # pip install tensorflow-probability==0.17.0  mới tương thích với tf 2.10
import cv2   # pip install opencv-python
import numpy as np
import seaborn as sns # pip install seaborn
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.regularizers  import L2, L1
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (Callback, CSVLogger, EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, ReduceLROnPlateau)
from sklearn.metrics import confusion_matrix, roc_curve # pip install scikit-learn

### DATASETS
train_direct = "Python_310/.tensorflow_gpu/assets/Human_emotions_datasets/Emotions Dataset/train"
val_direct = "Python_310/.tensorflow_gpu/assets/Human_emotions_datasets/Emotions Dataset/test"
CLASS_NAME = ["angry","happy","sad"]

CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
    "LEARNING_RATE": 0.001,
    "N_EPOCHS": 100,
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
# Lấy bộ dữ liệu từ thư mục đã phân loại sẵn
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

print(train_datasets) # Nếu các giá trị của CLASS NAME ko trùng với tên thưu mục thì sẽ xảy ra lỗi

# # trực quan hóa 1 batch ảnh khi chưa chuyển áp dụng hàm  preprocess cho từng ảnh
# Nếu đã áp dụng hàm preprocrss thì thay title bằng lệnh sau: plt.title(CONFIGURATION["CLASS_NAMES"][tf.argmax(labels[i], axis = 0).numpy()])

# plt.figure(figsize = (12,12))
# for images, labels in train_datasets.take(1):
#   for i in range(16):
#     ax = plt.subplot(4,4, i+1)
#     plt.imshow(images[i]/255.)
#     plt.title(CONFIGURATION["CLASS_NAMES"][labels[i].numpy()])
#     plt.axis("off")
# plt.show()


### KHÔNG TĂNG CƯỜNG DỮ LIỆU

# training_dataset = (
#     train_datasets
#     .prefetch(tf.data.AUTOTUNE)
# )

# validation_dataset = (
#     val_datasets
#     .prefetch(tf.data.AUTOTUNE)
# )

### HẾT KHÔNG TĂNG CƯỜNG DỮ LIỆU

### TĂNG CƯỜNG DỮ LIỆU
## Thử nghiệm trước khi tăng cường dữ liệu thì các chỉ số sẽ như nào (kém hơn) và sau đó hãy tăng cường dữ liệu
# Lưu ý khi xoay hình ảnh mặt người, chúng ta ko nên xoay 180 độ, bởi mặt người sẽ bị ngược( không tự nhiên), làm cho model càng kém hơn

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

### HIỂN THỊ HÌNH ẢNH SAU KHI ÁP DỤNG TĂNG CƯỜNG DỮ LIỆU ĐỂ XEM HÌNH ẢNH SAU TĂNG CƯỜNG CÓ BẤT THƯỜNG, HOẶC KHÔNG TỰ NHIÊN
plt.figure(figsize = (12,12))

for images, labels in training_dataset.take(1):
  for i in range(16):
    ax = plt.subplot(4,4, i+1)
    plt.imshow(images[i]/255.)
    plt.title(CONFIGURATION["CLASS_NAMES"][tf.argmax(labels[i], axis = 0).numpy()])
    plt.axis("off")
plt.show()

### HẾT TĂNG CƯỜNG DỮ LIỆU


### TẠO MODEL
# Chuẩn hóa đầu vào
resize_rescale_layers = tf.keras.Sequential([
       Resizing(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
       Rescaling(1./255),
])

lenet_model = tf.keras.Sequential(
    [
    # Đầu vào có thể bất kì hình dạng kích thước nào, miễn nó là hình ảnh màu (None, None, 3)
    InputLayer(input_shape = (None, None, 3), ),
    # Chuẩn hóa lại hình ảnh cho đúng định dạng đầu vào cảu mode bằng cách thay đổi kích thước, chuẩn hóa các giá trị
    resize_rescale_layers,

    Conv2D(filters = CONFIGURATION["N_FILTERS"] , kernel_size = CONFIGURATION["KERNEL_SIZE"], strides = CONFIGURATION["N_STRIDES"] , padding='valid',
          activation = 'relu',kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D (pool_size = CONFIGURATION["POOL_SIZE"], strides= CONFIGURATION["N_STRIDES"]*2),
    Dropout(rate = CONFIGURATION["DROPOUT_RATE"] ),

    Conv2D(filters = CONFIGURATION["N_FILTERS"] , kernel_size = CONFIGURATION["KERNEL_SIZE"], strides = CONFIGURATION["N_STRIDES"] , padding='valid',
          activation = 'relu',kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D (pool_size = CONFIGURATION["POOL_SIZE"], strides= CONFIGURATION["N_STRIDES"]*2),
    Dropout(rate = CONFIGURATION["DROPOUT_RATE"] ),

    Conv2D(filters = CONFIGURATION["N_FILTERS"] , kernel_size = CONFIGURATION["KERNEL_SIZE"], strides = CONFIGURATION["N_STRIDES"] , padding='valid',
          activation = 'relu',kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D (pool_size = CONFIGURATION["POOL_SIZE"], strides= CONFIGURATION["N_STRIDES"]*2),
    Dropout(rate = CONFIGURATION["DROPOUT_RATE"] ),

    Conv2D(filters = CONFIGURATION["N_FILTERS"]*2 + 4, kernel_size = CONFIGURATION["KERNEL_SIZE"], strides=CONFIGURATION["N_STRIDES"], padding='valid',
          activation = 'relu', kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D (pool_size = CONFIGURATION["POOL_SIZE"], strides= CONFIGURATION["N_STRIDES"]*2),

    Flatten(),

    Dense( CONFIGURATION["N_DENSE_1"], activation = "relu", kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    Dropout(rate = CONFIGURATION["DROPOUT_RATE"]),

    Dense( CONFIGURATION['N_DENSE_2'], activation = "relu", kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    Dropout(rate = CONFIGURATION["DROPOUT_RATE"]),

    Dense( CONFIGURATION['N_DENSE_3'], activation = "relu", kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),

    Dense(CONFIGURATION["NUM_CLASSES"], activation = "softmax"),

])

lenet_model.summary()

### HUẤN LUYỆN

loss_function = CategoricalCrossentropy()
# ví dụ về hàm loss CCE = -(y_true*log(y_predict))
print("\n")
y_true = [[0,1,0],[0,0,1]]
y_predict = [[0.05,0.95,0],[0.1,0.05,0.85]]
cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true=y_true, y_pred=y_predict).numpy())

metrics = [CategoricalAccuracy(name = "accuracy"), TopKCategoricalAccuracy(k = 2, name = "top_k_accuracy")]

lenet_model.compile(optimizer= Adam(learning_rate = CONFIGURATION["LEARNING_RATE"]),
                                loss = loss_function,
                                metrics = metrics,
                                ) # run_eagerly=True thêm tham số này khi trong huấn luyện có lỗi

# Lưu model theo từng giai đoạn

checkpoint_filepath = 'Python_310/.tensorflow_gpu/assets/save_model'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True) # chế độ ghi đè, chỉ lưu mỗi cái độ chính xác cao nhất

# # Lưu mỗi các trọng số, ko lưu model
# checkpoint_filepath = '/tmp/ckpt/checkpoint.weights.h5'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True)
# # Khi sử dụng thì phải dùng câu lệnh sau:
# model.load_weights(checkpoint_filepath)

history = lenet_model.fit(train_datasets, 
                          validation_data= validation_dataset,
                          epochs= CONFIGURATION["N_EPOCHS"],
                          verbose=1, 
                          callbacks=[model_checkpoint_callback])

### BIỂU DIỄN LOSS 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()

# Vẽ biểu đồ độ chính xác
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy'])
plt.show()

### KIỂM TRA HIỆU SUẤT CỦA MODEL
lenet_model.evaluate(validation_dataset)

### LƯU MODEL
# Một cách lưu và tải model thường được sử dụng ở tensorflow phiên bản 2.10 trở về, còn mới hơn thì phải thêm đuôi "Malaria_model_save.keras"
lenet_model.save("Human_emotions_detect")

### TỪ BƯỚC SAU SỬ DỤNG MÔ HÌNH ĐÃ LƯU
# Khởi tạo mô hình đã lưu
# lenet_model = tf.keras.models.load_model("Human_emotions_detect")
# lenet_model.summary()

# thử nghiệm 1 hình ảnh
test_image = cv2.imread("Python_310/.tensorflow_gpu/assets/Human_emotions_datasets/Emotions Dataset/test/happy/2801.jpg_rotation_1.jpg")
im = tf.constant(test_image, dtype = tf.float32)
print(im.shape)
im = tf.expand_dims(im, axis= 0)
print(im.shape)
print(lenet_model(im)) # tf.Tensor([[0.01408774, 0.9081586, 0.07775359]], shape=(1, 3), dtype=float32)
                            # vị trí:    0           1            2
print(tf.argmax(lenet_model(im), axis= 1)[0].numpy()) # Lấy vị trí của giá trị lớn nhất trong ma trận (1,3) theo axis = 1 (là theo cột)
print(CLASS_NAME[tf.argmax(lenet_model(im), axis= 1)[0].numpy()])

# Thử nghiệm 1 batch ảnh và show
plt.figure(figsize = (12,12))

for images, labels in val_datasets.take(1):
  for i in range(16):
    ax = plt.subplot(4,4, i+1)
    plt.imshow(images[i]/255.)

    true_label = CONFIGURATION["CLASS_NAMES"][tf.argmax(labels[i], axis = 0).numpy()]
    predict_by_model = CLASS_NAME[tf.argmax(lenet_model(tf.expand_dims(images[i], axis= 0)), axis= 1)[0].numpy()]

    plt.title("True Label: " + true_label + "\n"+ "Predict by model: " + predict_by_model)
    plt.axis("off")
plt.show()

# Ma trận nhầm lẫn
predicted = []
labels = []
for img, label in validation_dataset:
   predicted.append(lenet_model(img))
   labels.append(label.numpy())

# Nếu lấy các chỉ số index trong mảng predicted bằng np.argmax(predicted) thì sẽ gây ra lỗi
# Bởi bì predicted hiện đang theo các batch size: 32, ví dụ tổng có 98 phần tử thì batch1: 32, batch2:32, batch3:32 batch4:2
# Như vậy số lượng phần tử trong batch cuối sẽ không đủ 32 sẽ gây ra lỗi, vì vạy ta sẽ không lấy index của batch cuối
# hoặc nối 2 đoạn với nhau bằng câu lệnh sau: pre = np.concatenate([np.argmax(labels[:-1], axis = -1).flatten(), np.argmax(labels[-1], axis = -1).flatten()])
pre = np.argmax(predicted[:-1], axis= -1).flatten()
lab = np.argmax(labels[:-1], axis= -1).flatten()

cm = confusion_matrix(lab, pre)
print(cm)
plt.figure(figsize=(8,8))

sns.heatmap(cm, annot = True)
plt.title("Confusion matrix")
plt.ylabel("Actual")
plt.xlabel("Predict")
plt.show()