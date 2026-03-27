# RISE Internship – All Deep Learning Projects (Single File)

# ===============================

# IMPORTS

# ===============================

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# ===============================

# 1. IMAGE CLASSIFICATION (CNN)

# ===============================

print("\n🔹 Running Image Classification...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

cnn_model = models.Sequential([
layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
layers.MaxPooling2D(),
layers.Conv2D(64,(3,3),activation='relu'),
layers.MaxPooling2D(),
layers.Flatten(),
layers.Dense(64,activation='relu'),
layers.Dense(10,activation='softmax')
])

cnn_model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

cnn_model.fit(x_train, y_train, epochs=2, validation_data=(x_test,y_test))

# ===============================

# 2. SENTIMENT ANALYSIS (LSTM)

# ===============================

print("\n🔹 Running Sentiment Analysis...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

lstm_model = tf.keras.Sequential([
layers.Embedding(10000,128),
layers.LSTM(64),
layers.Dense(1,activation='sigmoid')
])

lstm_model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])

lstm_model.fit(x_train, y_train, epochs=2, validation_data=(x_test,y_test))

# ===============================

# 3. SPEECH RECOGNITION (FEATURE EXTRACTION)

# ===============================

print("\n🔹 Running Speech Feature Extraction...")

import librosa

# Use your own audio file path

# audio, sr = librosa.load('sample.wav')

# mfcc = librosa.feature.mfcc(y=audio, sr=sr)

# print("MFCC Shape:", mfcc.shape)

print("Speech module ready (add audio file to run)")

# ===============================

# 4. OBJECT DETECTION (YOLO - OPENCV)

# ===============================

print("\n🔹 Object Detection Setup...")

import cv2

# Uncomment after adding YOLO files

# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

print("YOLO ready (add weights + image to run)")

# ===============================

# 5. RECOMMENDATION SYSTEM

# ===============================

print("\n🔹 Running Recommendation System...")

num_users = 1000
num_items = 500

user_input = layers.Input(shape=(1,))
item_input = layers.Input(shape=(1,))

user_embedding = layers.Embedding(num_users,50)(user_input)
item_embedding = layers.Embedding(num_items,50)(item_input)

x = layers.concatenate([user_embedding,item_embedding])
x = layers.Flatten()(x)
x = layers.Dense(128,activation='relu')(x)
output = layers.Dense(1)(x)

rec_model = tf.keras.Model([user_input,item_input], output)
rec_model.compile(optimizer='adam', loss='mse')

print(rec_model.summary())

# ===============================

# 6. ANOMALY DETECTION (AUTOENCODER)

# ===============================

print("\n🔹 Running Anomaly Detection...")

input_dim = 20

autoencoder = tf.keras.Sequential([
layers.Dense(16,activation='relu',input_shape=(input_dim,)),
layers.Dense(8,activation='relu'),
layers.Dense(16,activation='relu'),
layers.Dense(input_dim,activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

data = np.random.rand(1000, input_dim)
autoencoder.fit(data, data, epochs=2)

# ===============================

# 7. TIME SERIES FORECASTING (LSTM)

# ===============================

print("\n🔹 Running Time Series Forecasting...")

data = np.sin(np.arange(0,100,0.1))

def create_seq(data, window=10):
X, y = [], []
for i in range(len(data)-window):
X.append(data[i:i+window])
y.append(data[i+window])
return np.array(X), np.array(y)

X, y = create_seq(data)

X = X.reshape((X.shape[0], X.shape[1], 1))

ts_model = tf.keras.Sequential([
layers.LSTM(50, input_shape=(10,1)),
layers.Dense(1)
])

ts_model.compile(optimizer='adam', loss='mse')
ts_model.fit(X, y, epochs=2)

# ===============================

# 8. END-TO-END PIPELINE

# ===============================

print("\n🔹 Running End-to-End Pipeline...")

def preprocess(data):
return data / 255.0

def predict(model, sample):
sample = preprocess(sample)
return model.predict(sample)

print("Pipeline Ready ")

# ===============================

# DONE

# ===============================

print("\n ALL PROJECTS EXECUTED SUCCESSFULLY!")
