import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds  # pip install tensorflow-datasets

#model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization


# Tải dataset và thông tin của dataset
dataset, dataset_info = tfds.load("malaria", with_info=True, split=["train[:80%]", "train[80%:90%]", "train[90%:]"], shuffle_files=True)

# In từng mẫu dataset của bộ huấn luyện, thử nghiệm với một mẫu
for data in dataset[0].take(1):
  print(data)

# Bộ dữ liệu được chia thành 3 phần, khi print(dataset) ta sẽ thấy được 3 list trong dataset
train_dataset = dataset[0]
val_dataset = dataset[1]
test_dataset = dataset[2]

# Hàm chuyển đổi ảnh sang float32
def convert_image(data):
    image = tf.cast(data['image'], tf.float32)  # Chuyển đổi ảnh sang float32
    label = data['label']
    return {'image': image, 'label': label}

# Định dạng đầu vào là 224 bởi có một số hình ảnh không đồng nhất nên lấy 224 làm mốc chung cho đầu vào
new_shape = 224
def resize_image(dataset):
   return tf.image.resize(dataset["image"], (new_shape, new_shape))/255., dataset["label"]

# # Áp dụng chuyển đổi cho dataset huấn luyện
# train_dataset = train_dataset.map(convert_image)
# val_dataset = val_dataset.map(convert_image)
# test_dataset = test_dataset.map(convert_image)

# Hiển thị 16 ảnh đầu tiên trong dataset huấn luyện
plt.figure(figsize=(10, 10))
for i, data in enumerate(train_dataset.take(16)):
    image, label = data['image'], data['label']
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(image.numpy())  # Hiển thị ảnh dưới dạng uint8
    # chuyển đổi dạng tensor của label thành các ký tự có thể đọc được
    plt.title(dataset_info.features['label'].int2str(label.numpy()))
    plt.axis('off')
plt.show()

# ĐẦu vào sẽ có hình dạng là (x1,x2,x3,x4) theo thứ tự là batch size, height, width, channels (hình ảnh gray thì channels =1, hình ảnh màu thì channels = 3)
# Thêm batch size cho hình ảnh đầu vào làm dữ liệu huấn luyện

train_dataset = train_dataset.map(resize_image).shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(resize_image).shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(resize_image).shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

for image, label in train_dataset.take(1):
   print(image, label)

# Tạo một model tuần tự với các lớp theo thứ tự
# 1 lớp tích chập 2D có 6 bộ lọc với kích thước mỗi bộ lọc là 3x3, bước nhảy là một, hàm kích hoạt là relu và có đầu vào là hình ảnh màu (224,224,3)
# Tiếp theo là lớp maxpoling nhằm giamrnkichs thước, với kích thước 2x2 và bước nhảy là 2
model = tf.keras.Sequential([
   Conv2D(6, (3,3), strides = 1, padding = 'valid',  activation = 'relu', input_shape = (224,224,3)),
   # chuẩn hóa hàng loạt
   BatchNormalization(),
   MaxPool2D(pool_size=(2, 2),strides= 2),

   Conv2D(10, (3,3), strides = 1, padding = 'valid',  activation = 'relu'),
   BatchNormalization(),
   MaxPool2D(pool_size=(2, 2),strides= 2),
    # Làm phẳng các tính năng thu được để đưa vào lớp Dense
   Flatten(),
   Dense(50  , activation = "relu"),
   BatchNormalization(),
   Dense(10, activation = "relu"),
   BatchNormalization(),
   Dense(1, activation = "sigmoid")
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)
# Huấn luyện mô hình
epoch_hits = model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1)
# Đánh giá mô hình với tập thử nghiệm
print(model.evaluate(test_dataset))

# Dự đoán thử
# Hàm take sẽ lấy 1 mẫu từ bộ dữ liệu, tuy nhiên bộ dữ liệu hiện tại đang có hình dạng là (batch, height, width, channels) nên take (1) sẽ không lấy một dữ liệu đơn lẻ mà sẽ là 1 batch gồm 32 dữ liệu đơn lẻ
# vì vậy nó sẽ trả về 32 dự đoán chứa trong 1 tensor 2D
print("Prediction:", model.predict(test_dataset.take(1)))
# Cách dự đoán từng mẫu đơn lẻ
# print("Prediction:",model.predict(test_dataset.take(1)))[0][0]
# print("True label:", label.numpy()[0])
# hoặc lấy một mẫu từ tập kiểm tra
single_sample = test_dataset.unbatch().take(1)

# Chuyển đổi single_sample sang dạng batch với batch size = 1
single_sample = single_sample.batch(1)

# Dự đoán cho một mẫu duy nhất
for image, label in single_sample:
    prediction = model.predict(image)
    print("Prediction:", prediction)
    print("True label:", label.numpy())

# Lưu model cùng với các thông số đã huấn luyện hoặc chỉ với các thông số đã huấn luyện
export_path = ".tensorflow_course/assests/Malaria_model"
tf.saved_model.save(model, export_dir=export_path)
# Tải model với các trọng số đã lưu
load_model = tf.saved_model.load(
    export_path, tags=None, options=None
)
# Không thể gọi summary đối với các model lưu bằng savemodel
# load_model.summary()
# Kiểm tra các signature có sẵn trong mô hình đã tải
print(load_model.signatures.keys())

# Lấy default signature để sử dụng mô hình đã tải, làm việc và dự đoán với tên model mới là inter
infer = load_model.signatures["serving_default"]

# Một cách lưu và tải model khác thường được sử dụng
model.save("Malaria_model_save.keras")

load_save_model = tf.keras.models.load_model("Malaria_model_save.keras")
load_save_model.summary()