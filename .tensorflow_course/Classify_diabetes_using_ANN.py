import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# print(tf.__version__)
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Feature Scaling is a must in ANN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Google colab: https://colab.research.google.com/drive/1Ig3XuHBid7EtCshQCrtd5OhYo1xkTyNC
diabetes_df = pd.read_csv(".tensorflow_course\\assests\\diabetes.csv")
# print(diabetes_df)
# print(diabetes_df.describe())
# print(diabetes_df.info())

### LÀM SẠCH DỮ LIỆU VÀ HIỂN THỊ
# ĐẾm và hiển thị số lượng của các phần tử trong cột Outcome
sns.countplot(x = 'Outcome', data = diabetes_df)
plt.show()

# sns.pairplot(diabetes_df, hue = 'Outcome', vars = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
# plt.show()

# sns.heatmap(diabetes_df.corr(), annot = True)
# plt.show()
## Lấy dữ liệu thành 2 loại là input và output
X = diabetes_df.iloc[:, 0:8].values
y = diabetes_df.iloc[:, 8].values
# print(y)
# Thực hiện đồng bộ đầu vào x thành các vecto
sc = StandardScaler()
X = sc.fit_transform(X)

# Chia bộ dữ liệu thành training và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

### HUẤN LUYỆN MÔ HÌNH
# Tạo mô hình với 2 lớp ẩn cùng với 2 hàm dropout để loại bỏ ngẫu nhiên các node trong quá trình thực hiện huấn luyện mô hình với xác suất là 0.2
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units=400, activation='relu', input_shape=(8, )))
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Dense(units=400, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# classifier.summary()

classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
epochs_hist = classifier.fit(X_train, y_train, epochs = 200)

# Dự đoán với bộ dữ liệu test
y_pred = classifier.predict(X_test)
print(y_pred)
# Chuyển đổi các giá trị của y về False nếu kết quả của nó thấp hơn 0.5 và chuyển thành true nếu nó lớn hơn 0.5
y_pred = (y_pred > 0.5)
print(y_pred)

### ĐÁNH GIÁ MÔ HÌNH
print(epochs_hist.history.keys())
# Hiển thị giá trị loss trong quá trình huấn luyện
plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss'])
plt.show()

# Hiển thị ma trận nhầm lẫn
y_train_pred = classifier.predict(X_train)
y_train_pred = (y_train_pred > 0.5)
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot=True)
plt.show()

# Hiển thị các giá trị còn lại như precision, recall, support, nếu các giá trị bằng 1 thì khả năng cao mô hình đã overfit
print(classification_report(y_train_pred, y_train))

# Hiển thị ma trận nhầm lẫn của y test và y dự đoán
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_pred))