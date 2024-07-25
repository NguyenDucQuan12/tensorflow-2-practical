import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds  # pip install tensorflow-datasets


from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, InputLayer, Dropout
from tensorflow.keras.regularizers import L1, L2
from tensorboard.plugins.hparams import api as hp

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
# Không sử dụng cá tham số cài đặt sẵn
# DROPOUT_RATE = 0.3
# REGULARIZATION_RATE = 0.2
def create_model_auto_hyperparam(hparams):
    model = tf.keras.Sequential([
    InputLayer(shape = (224,224,3)),
    Conv2D(6, (3,3), strides = 1, padding = 'valid',  activation = 'relu', kernel_regularizer = L2(hparams["HP_REGULARIZATION_RATE"])),
    # chuẩn hóa hàng loạt
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2),strides= 2),
    Dropout(rate = hparams["HP_DROPOUT"]),

    Conv2D(10, (3,3), strides = 1, padding = 'valid',  activation = 'relu',kernel_regularizer = L2(hparams["HP_REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2),strides= 2),
        # Làm phẳng các tính năng thu được để đưa vào lớp Dense
    Flatten(),
    Dense(hparams["HP_NUM_UNITS_1"]  , activation = "relu", kernel_regularizer = L2(hparams["HP_REGULARIZATION_RATE"])),
    BatchNormalization(),
    Dropout(rate = hparams["HP_DROPOUT"]),
    Dense(hparams["HP_NUM_UNITS_2"], activation = "relu", kernel_regularizer = L2(hparams["HP_REGULARIZATION_RATE"])),
    BatchNormalization(),
    Dense(1, activation = "sigmoid")
    ])

    model.summary()

    # Nếu trong quá trình huấn luyện mà găp phải lỗi thì thêm tham số sau đây: run_eagerly = True vào model.compile để thông báo lỗi chi tiết hơn


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = hparams["HP_LEARNING_RATE"]),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    # Huấn luyện mô hình
    epoch_hits = model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1)

    # Đánh giá mô hình với tập thử nghiệm
    _, accuracy = model.evaluate(test_dataset)

    return accuracy

# Các siêu tham số sẽ được chọn trong khoảng bên dưới
# Siêu tham số là các giá trị được khởi tạo ban đầu, không được học hay thay đổi trong quá trình training hay predict
HP_NUM_UNITS_1 = hp.HParam("num_units_1", hp.Discrete([16,32,64,128]))
HP_NUM_UNITS_2 = hp.HParam("num_units_2", hp.Discrete([16,32,64,128]))
HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.1,0.2,0.3]))
HP_REGULARIZATION_RATE = hp.HParam("regularization_rate", hp.Discrete([0.001,0.01,0.1]))
HP_LEARNING_RATE = hp.HParam("learning_rate", hp.Discrete([1e-4,1e-3]))

run_number = 0
for num_units_1 in HP_NUM_UNITS_1.domain.values:
    for num_units_2 in HP_NUM_UNITS_2.domain.values:
        for dropout_rate in HP_DROPOUT.domain.values:
            for regularization_rate in HP_REGULARIZATION_RATE.domain.values:
                for learning_rate in HP_LEARNING_RATE.domain.values:

                    hparams = {
                        "HP_NUM_UNITS_1": num_units_1,
                        "HP_NUM_UNITS_2": num_units_2,
                        "HP_DROPOUT": dropout_rate,
                        "HP_REGULARIZATION_RATE": regularization_rate,
                        "HP_LEARNING_RATE": learning_rate,

                    }

                    file_writer = tf.summary.create_file_writer("logs/hparams-"+ str(run_number))
                    
                    print("for run {}: Our hparams is HP_NUM_UNITS_1 - {}, HP_NUM_UNITS_2 - {}, HP_DROPOUT - {}, HP_REGULARIZATION_RATE - {}, HP_LEARNING_RATE - {}".format(run_number,\
                     hparams["HP_NUM_UNITS_1"], hparams["HP_NUM_UNITS_2"], hparams["HP_DROPOUT"],hparams["HP_REGULARIZATION_RATE"],hparams["HP_LEARNING_RATE"]))

                    with file_writer.as_default():
                        hp.hparams(hparams)
                        accuracy = create_model_auto_hyperparam(hparams= hparams)
                        tf.summary.scalar("accuracy", accuracy, step = 0)

                    
                    run_number = run_number+1

### Sau khi chọn được các tham số làm cho mô hình có độ chính xác cao thì thay thế các tham số đó vào mô hình chính thức
### Để chi tiết hơn thì tìm hiểu cách thức xem trực quan hóa dữ liệu bằng tensorboard
