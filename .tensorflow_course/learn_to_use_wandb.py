import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds  # pip install tensorflow-datasets

#model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, InputLayer, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC
# pip install wandb
import wandb
from wandb.integration.keras import WandbMetricsLogger


wandb.init(project="test_project", entity="test_my_project",
           config={
                    "learning_rate": 0.001,
                    "n_epochs": 100,
                    "batch_size": 128,
                    "dropout_rate": 0.0,
                    "IM_size":224,
                    "regularization_rate": 0.0,
                    "n_filler":6,
                    "kernel_size": 3,
                    "n_strides":1,
                    "pool_size": 2,
                    "n_dense_1": 128,
                    "n_dense_1":32 
                }
)

# Tải dataset và thông tin của dataset, shuffle_files=True thì mỗi lần truy cập dữ liệu thì nó sẽ trộn ngẫu nhiên
dataset, dataset_info = tfds.load("malaria", with_info=True, split=["train[:80%]", "train[80%:90%]", "train[90%:]"], shuffle_files=True)

# Bộ dữ liệu được chia thành 3 phần, khi print(dataset) ta sẽ thấy được 3 list trong dataset
train_dataset = dataset[0]
val_dataset = dataset[1]
test_dataset = dataset[2]




#### CÓ 2 CÁCH ĐỂ THAY ĐỔI KÍCH THƯỚC VÀ TĂNG CƯỜNG DỮ LIỆU Ở TENSORFLOW, SỬ DỤNG CÁCH NÀO CŨNG ĐƯỢC
### TUY NHIÊN TF.IMAGE CÓ NHIỀU LỰA CHỌN HƠN
# Định dạng đầu vào là 224 bởi có một số hình ảnh không đồng nhất nên lấy 224 làm mốc chung cho đầu vào
new_shape = 224

# Sử dụng lớp image của tensorflow để thay đổi kích thước và chuẩn hóa data nằm trong khoảng [0,1]
def resize_image(dataset):
   return tf.image.resize(dataset["image"], (new_shape, new_shape))/255.0, dataset["label"]

# Sử dụng lớp layer của tensorflow để thay đổi kích thước và chuẩn hóa data nằm trong khoảng [0,1]
resize_rescale_layer = tf.keras.Sequential([
   tf.keras.layers.Resizing(new_shape, new_shape),
   tf.keras.layers.Rescaling(1.0/255),
])
def resize_image_with_layer(dataset):
    return resize_rescale_layer(dataset["image"]), dataset["label"]

# tăng cường dữ liệu với lớp image của tensorflow
def augment(datasets) :
   image, label = resize_image(datasets)
   image = tf.image.rot90(image)
#    image = tf.image.adjust_saturation(image=image, saturation_factor= 0.3)
   image = tf.image.flip_left_right(image)
   return image, label 


# tăng cường dữ liệu với lớp layer của tensorflow
augment_with_layer = tf.keras.Sequential([
   tf.keras.layers.RandomFlip(mode = "horizontal",),
   tf.keras.layers.RandomRotation(factor = (0.25,0.2501),)
])

def augment_layer(datasets):
   image, label = resize_image(datasets)
   image = augment_with_layer(image, training = True)
   return image, label





# Trực quan hóa hình ảnh gốc và ảnh được tăng cường
def visualize(original_img, augmented_img):
   # Định dạng ô chứa là 1 hàng 2 cột, và hình ảnh thứ nhất sẽ nằm ở ô thứ 1
   plt.subplot(1,2,1) 
   plt.imshow(original_img)

   plt.subplot(1,2,2)
   plt.imshow(augmented_img)

   plt.show()

# Lấy phần tử đầu tiên của bộ dữ liệu huấn luyện, mỗi lần gọi next(iter(train_dataset)) sẽ lấy phần tử tiếp theo
# Iter sẽ tạo một phần tử của bộ dữ liệu iterable (là bộ dữ liệu có thể lặp lại từng phần tử), kết quả là 1 dictionary gồm 2 khóa "image" và "label"
data_1 = next(iter(train_dataset))
original_img = data_1["image"]
augmented_img = tf.image.flip_left_right(original_img)
# augmented_img = tf.image.random_flip_up_down(original_img)
# augmented_img = tf.image.rot90(original_img)
# augmented_img = tf.image.adjust_brightness(original_img, delta=0.8)

visualize(original_img, augmented_img) 

# Cẩn thận khi lựa chọn các phương pháp tăng cường, bởi nếu chọn bừa bãi sẽ làm cho dữ liệu mất đi sự khác biệt giữa các loại nhã
# Ví dụ như bộ dữ liệu ung thư, nếu áp dụng độ bão hòa thì hình ảnh ung thư và không ung thư sẽ không có sự khác nhau

# plt.figure(figsize=(10, 10))
for i, data in enumerate(train_dataset.take(4)):
    image, label = data['image'], data['label']
    plt.subplot(2, 4, 2*i + 1)
    plt.imshow(image)  # Hiển thị ảnh dưới dạng uint8
    # chuyển đổi dạng tensor của label thành các ký tự có thể đọc được
    plt.title(dataset_info.features['label'].int2str(label.numpy()))

    plt.subplot(2, 4, 2*i + 2)
    plt.imshow(tf.image.adjust_brightness(image, delta=0.8))  # Hiển thị ảnh dưới dạng uint8
    # chuyển đổi dạng tensor của label thành các ký tự có thể đọc được
    plt.title(dataset_info.features['label'].int2str(label.numpy()))

    plt.axis('off')
plt.show()


# ĐẦu vào sẽ có hình dạng là (x1,x2,x3,x4) theo thứ tự là batch size, height, width, channels (hình ảnh gray thì channels =1, hình ảnh màu thì channels = 3)
# Hàm map sẽ áp dụng hàm resize_image cho mỗi phần tử của bộ dữ liệu về 224x224
# hàm shuffle sẽ trộn dữ liệu ngẫu nhiên với nhau với bộ đệm = 8, và trộn dữ liệu sau mỗi epoch đảm bảo sau mỗi lần huấn luyện sẽ được trộn
# batch sẽ gộp các dữ liệu thành các batch chứa 32 phần tử
# hàm prefetch được sử dụng để tải trước các batch dữ liệu tiếp theo trong khi mô hình đang huấn luyện trên các batch hiện tại, giúp tăng tốc quá trình huấn luyện
# Tham số tf.data.AUTOTUNE: Tự động điều chỉnh số lượng batch được tải trước dựa trên tài nguyên hệ thống (CPU và RAM)
###  SAU KHI CHẠY HÀM BÊN DƯỚI THÌ MỖI LẦN TRUY CẬP CÁC PHẦN TỬ CỦA BỘ DỮ LIỆU THÌ NÓ TRỘN NGẪU NHIÊN, VÀ ÁP DỤNG LẦN LƯỢT CÁC PHƯƠNG PHÁP TĂNG CƯỜNG DATA
# Muốn tăng dữ liệu thì phải lưu các hình ảnh dã tăng cường vào một bộ lưu trữ khác, còn ở đây mỗi khi truy cập nó sẽ tạo ra 1 bản sao lưu mới đã tăng cường 
# và ta sẽ sử dụng bản sao lưu đó, tức là ta truy cập data thì nó tạo ra 1 hình ảnh mơi đã được tăng cường, ảnh gốc vẫn giữ nguyên, nếu muốn truy cập ảnh gốc
# thì phải truy cập trước khi chạy hàm bên dưới hoặc hủy bỏ các lệnh tăng cường
train_dataset = (train_dataset
                 .shuffle(buffer_size = 8, reshuffle_each_iteration = True)
                 .map(augment_layer)
                 .batch(32)
                 .prefetch(tf.data.AUTOTUNE))

val_dataset = (val_dataset
               .shuffle(buffer_size = 8, reshuffle_each_iteration = True)
               .map(augment)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE))


test_dataset = (test_dataset
                .shuffle(buffer_size = 8, reshuffle_each_iteration = True)
                .map(augment)
                .batch(32)
                .prefetch(tf.data.AUTOTUNE))


# Tạo một model tuần tự với các lớp theo thứ tự
# 1 lớp tích chập 2D có 6 bộ lọc với kích thước mỗi bộ lọc là 3x3, bước nhảy là một, hàm kích hoạt là relu và có đầu vào là hình ảnh màu (224,224,3)
# Tiếp theo là lớp maxpooling nhằm giảm kích thước, với kích thước 2x2 và bước nhảy là 2

configuration = wandb.config
IM_SIZE = configuration["IM_size"]
DROPOUT_RATE = configuration["dropout_rate"]
REGULARIZATION_RATE = configuration["regularization_rate"]
N_FILTERS = configuration["n_filler"]
KERNEL_SIZE = configuration["kernel_size"]
POOL_SIZE = configuration["pool_size"]
N_STRIDES = configuration["n_strides"]



model = tf.keras.Sequential([
   InputLayer(shape = (IM_SIZE,IM_SIZE,3)),
   Conv2D(filters = N_FILTERS, kernel_size = KERNEL_SIZE, strides = N_STRIDES, padding = 'valid',  activation = 'relu', kernel_regularizer = L2(REGULARIZATION_RATE)),
   # chuẩn hóa hàng loạt
   BatchNormalization(),
   MaxPool2D(pool_size=POOL_SIZE, strides= N_STRIDES*2),
   Dropout(rate = DROPOUT_RATE),

   Conv2D(filters = N_FILTERS*2+4, kernel_size = KERNEL_SIZE, strides = N_STRIDES, padding = 'valid',  activation = 'relu',kernel_regularizer = L2(REGULARIZATION_RATE)),
   BatchNormalization(),
   MaxPool2D(pool_size= POOL_SIZE, strides= N_STRIDES*2),
    # Làm phẳng các tính năng thu được để đưa vào lớp Dense
   Flatten(),
   Dense(50  , activation = "relu"),
   BatchNormalization(),
   Dropout(rate = DROPOUT_RATE),
   Dense(32, activation = "relu"),
   BatchNormalization(),
   Dense(1, activation = "sigmoid")
])

model.summary()

# Nếu trong quá trình huấn luyện mà găp phải lỗi thì thêm tham số sau đây: run_eagerly = True vào model.compile để thông báo lỗi chi tiết hơn

metrics = [TruePositives(name = "TP"), FalsePositives(name = "FP"), TrueNegatives(name = "TN"), FalseNegatives(name = "FN"),
           BinaryAccuracy(name = "Accuracy"), Precision(name = "Precision"), Recall(name = "Recall"), AUC(name= "AUC")]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate= configuration["learning_rate"]),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=metrics,
)
# Huấn luyện mô hình
epoch_hits = model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1, callbacks = [WandbMetricsLogger()])

# Đánh giá mô hình với tập thử nghiệm
print(model.evaluate(test_dataset))


# Cách dự đoán từng mẫu đơn lẻ
single_sample = test_dataset.unbatch().take(1)
single_sample = single_sample.batch(1)

# Dự đoán cho một mẫu duy nhất
for image, label in single_sample:
    prediction = model.predict(image)
    print("Prediction:", prediction)
    print("True label:", label.numpy())

# Một cách lưu và tải model khác thường được sử dụng
model.save("Malaria_model_save.keras")
load_save_model = tf.keras.models.load_model("Malaria_model_save.keras")
load_save_model.summary()

wandb.finish()