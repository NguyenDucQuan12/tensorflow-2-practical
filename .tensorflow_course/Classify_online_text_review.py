import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# print(tf.__version__)
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix




"""
https://colab.research.google.com/drive/13G61ZaSpzBz3T8x5SRkWCkJO8kGCvjeP#scrollTo=xNl52nl3qiyL
Dataset consists of 3000 Amazon customer reviews, star ratings, date of review, variant and feedback of various amazon Alexa products like Alexa Echo, Echo dots.
The objective is to discover insights into consumer reviews and perfrom sentiment analysis on the data.
Dataset: www.kaggle.com/sid321axn/amazon-alexa-reviews

Nếu feedback của khách hàng được đánh giá là 1 thì có nghĩa đấy là feedback tốt, còn 0 thì sẽ là đánh giá không tốt
"""
alexa_df = pd.read_csv(".tensorflow_course\\assests\\amazon_alexa.tsv", sep= '\t')
# print(alexa_df)

### TRỰC QUAN HÓA DỮ LIỆU
# Lấy các hàng có feedback là 1
positive = alexa_df[alexa_df['feedback']==1 ]
# print(positive)

negative = alexa_df[alexa_df['feedback']==0 ]

# sns.countplot(alexa_df['feedback'], label = "Count")
# plt.show()
# alexa_df['rating'].hist(bins= 5)
# plt.show()

### LÀM SẠCH DỮ LIỆU
# Loại bỏ một số cột không cần thiết, hoặc giảm thiểu độ phức tạp cảu bài toán
alexa_df = alexa_df.drop(['date','rating'], axis=1)

variation_dummies = pd.get_dummies(alexa_df['variation'].apply(lambda x: np.str_(x)), drop_first= True)
variation_dummies = variation_dummies.astype(int)
# print(variation_dummies)
alexa_df.drop(['variation'], axis= 1, inplace= True)
# print(alexa_df)
# Ghép 2 dataset lại để thành 1 bảng mới hoàn chỉnh
alexa_df = pd.concat([alexa_df, variation_dummies], axis= 1)
# print(alexa_df)
## COUNT VECTORIZER
# Tách từng chữ trong đoạn dữ liệu "sample_data" và so sánh xem chữ đó có tồn tại trong câu không và số lượng bao nhiêu
# sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(sample_data)
# print(X)
# print(vectorizer.get_feature_names_out())

# print(alexa_df['verified_reviews'])
vectorizer = CountVectorizer()
# Vì đối tượng của một cột trong dataFrame là object (dtype: object) nên cần chuyển đổi nó thành unicode trước khi tách các chữ 
#Khi sử dụng values.astype("U") có thể gây tràn bộ nhớ với các file lớn (80000 dòng), thay vào đó có thể dùng apply(lambda x: np.str_(x))) hoặc astype('U').values
alexa_count_vect = vectorizer.fit_transform(alexa_df['verified_reviews'].apply(lambda x: np.str_(x)))
# print(alexa_count_vect.toarray())
# print(vectorizer.get_feature_names_out())

# first let's drop the column
alexa_df.drop(['verified_reviews'], axis=1, inplace=True)
reviews = pd.DataFrame(alexa_count_vect.toarray())
# Now let's concatenate them together
alexa_df = pd.concat([alexa_df, reviews], axis=1)
# Tạo các giá trị đầu ra đầu vào
X = alexa_df.drop(['feedback'], axis= True)
Y = alexa_df['feedback']

### TẠO MODEL VÀ HUẤN LUYỆN MODEL
# Phân chia giá trị test và train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state=5)
# print(y_train.shape)
# print(y_test.shape)

# Tạo model với 2 lóp ẩn
ANN_classifier = tf.keras.models.Sequential()
ANN_classifier.add(tf.keras.layers.Dense(units= 400, activation = 'relu', input_shape = (4060,)))
ANN_classifier.add(tf.keras.layers.Dense(units= 400, activation = 'relu'))
ANN_classifier.add(tf.keras.layers.Dense(units= 1, activation = 'sigmoid'))

# Xem kết quả model
# ANN_classifier.summary()

# Huấn luyện mô hình
ANN_classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
epochs_hits = ANN_classifier.fit(X_train, y_train, epochs = 10)

# Đánh giá mô hình
y_predict = ANN_classifier.predict(X_test)
# print(y_predict )
# chuyển đổi các giá trị số thành các giá trị True hoặc False
y_predict = (y_predict>0.5)

cm =confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot= True)
plt.show()

plt.plot(epochs_hits.history['loss'])
plt.title("Loss during training")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.show()