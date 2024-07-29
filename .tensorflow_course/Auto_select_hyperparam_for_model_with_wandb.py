import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import tensorflow as tf
import tensorflow_datasets as tfds  # pip install tensorflow-datasets

#model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, InputLayer, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC
# pip install wandb
import wandb
from wandb.integration.keras import WandbMetricsLogger


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

# sử dụng wandb
wandb.login()

sweep_configuration = {
    "name": "Malaria_config",
    "method": "random",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "LEARNING_RATE": {"distribution":"uniform","min": 0.0001, "max": 0.1},

        "N_DENSE_1": {"values": [16, 32, 64, 128]},

        "N_DENSE_2": {"values": [16, 32, 64, 128]},

        "DROPOUT_RATE": {"distribution":"uniform","min": 0.1, "max": 0.4},

        "REGULARIZATION_RATE": {"distribution":"uniform","min": 0.001, "max": 0.1},

        "BATCH_SIZE": {"values": [16, 32, 64]},

        "N_EPOCHS": {"values": [5, 10, 15]},

        "OPTIMIZER": {"values": ["adam", "sgd"]},
    },
} 

sweep_id = wandb.sweep(sweep=sweep_configuration, project="test_project")

CONFIGURATION={
                    "LEARNING_RATE": 0.001,
                    "N_EPOCHS": 3,
                    "BATCH_SIZE": 128,
                    "DROPOUT_RATE": 0.0,
                    "IM_SIZE":224,
                    "REGULARIZATION_RATE": 0.2,
                    "N_FILTER":6,
                    "KERNEL_SIZE": 3,
                    "N_STRIDES":1,
                    "POOL_SIZE": 2,
                    "N_DENSE_1": 128,
                    "N_DENSE_2":32 
                }

# Tạo một model tuần tự với các lớp theo thứ tự
# 1 lớp tích chập 2D có 6 bộ lọc với kích thước mỗi bộ lọc là 3x3, bước nhảy là một, hàm kích hoạt là relu và có đầu vào là hình ảnh màu (224,224,3)
# Tiếp theo là lớp maxpooling nhằm giảm kích thước, với kích thước 2x2 và bước nhảy là 2

def create_model_auto_hyperparam(config):
    model = tf.keras.Sequential([
    InputLayer(shape = (config.IM_SIZE, config.IM_SIZE,3)),

    Conv2D(filters = config.N_FILTER, kernel_size = config.KERNEL_SIZE, strides = config.N_STRIDES, padding = 'valid',
             activation = 'relu', kernel_regularizer = L2(config.REGULARIZATION_RATE)),
    # chuẩn hóa hàng loạt
    BatchNormalization(),
    MaxPool2D(pool_size=config.POOL_SIZE, strides= config.N_STRIDES),
    Dropout(rate = config.DROPOUT_RATE),

    Conv2D(filters = config.N_FILTER, kernel_size = config.KERNEL_SIZE, strides = config.N_STRIDES, padding = 'valid',
             activation = 'relu', kernel_regularizer = L2(config.REGULARIZATION_RATE)),
    BatchNormalization(),
    MaxPool2D(pool_size=config.POOL_SIZE, strides= config.N_STRIDES),

    # Làm phẳng các tính năng thu được để đưa vào lớp Dense
    Flatten(),

    Dense(config.N_DENSE_1 , activation = "relu", kernel_regularizer = L2(config.REGULARIZATION_RATE)),
    BatchNormalization(),
    Dropout(rate = config.DROPOUT_RATE),

    Dense(config.N_DENSE_2, activation = "relu", kernel_regularizer = L2(config.REGULARIZATION_RATE)),
    BatchNormalization(),

    Dense(1, activation = "sigmoid")
    ])

    model.summary()
    return model

def train():
    with wandb.init(project="test_project", entity="test_my_project", config = CONFIGURATION) as run:
        config = wandb.config
        model = create_model_auto_hyperparam(config= config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate= config.LEARNING_RATE),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics,
        )
        model.fit(train_dataset, validation_data = val_dataset,
                                epochs = config.N_EPOCHS , verbose = 1, callbacks = [WandbMetricsLogger()])


# Nếu trong quá trình huấn luyện mà găp phải lỗi thì thêm tham số sau đây: run_eagerly = True vào model.compile để thông báo lỗi chi tiết hơn

metrics = [TruePositives(name = "TP"), FalsePositives(name = "FP"), TrueNegatives(name = "TN"), FalseNegatives(name = "FN"),
           BinaryAccuracy(name = "Accuracy"), Precision(name = "Precision"), Recall(name = "Recall"), AUC(name= "AUC")]


# Huấn luyện mô hình với lần chạy là 5, nếu không nó sẽ không tự động dừng
count = 5
wandb.agent(sweep_id, function = train, count = count)

wandb.finish()