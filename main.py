import numpy as np
import pandas as pd
import random
import os
# import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import plot_model
from sklearn.metrics import classification_report
from collections import Counter
import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import Model, layers
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Input, Conv2D, Flatten,MaxPooling3D
from keras.layers import Conv2D, MaxPooling2D, Flatten,  BatchNormalization, Activation
from keras.preprocessing import image

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root_path = 'Project_CNN/'
train_pred_test_folders = os.listdir(root_path)
seg_train_folders = 'Project_CNN/seg_train/'
seg_test_folders = 'Project_CNN/seg_test/'
seg_pred_folders = 'Project_CNN/seg_pred/'
number_tra = {}
number_tes = {}
for folder in os.listdir(seg_train_folders):
    number_tra[folder] = len(os.listdir(seg_train_folders + folder))

for folder in os.listdir(seg_test_folders):
    number_tes[folder] = len(os.listdir(seg_test_folders + folder))

number_train = pd.DataFrame(list(number_tra.items()), index=range(0, len(number_tra)), columns=['class', 'count'])
number_test = pd.DataFrame(list(number_tes.items()), index=range(0, len(number_tes)), columns=['class', 'count'])

figure, ax = plt.subplots(1, 2, figsize=(20, 6))
sns.barplot(x='class', y='count', data=number_train, ax=ax[0])
sns.barplot(x='class', y='count', data=number_test, ax=ax[1])

print("Number of images in the train set : ", sum(number_tra.values()))
print("Number of images in the test set ; ", sum(number_tes.values()))
number_of_images_in_prediction_set = len(os.listdir(seg_pred_folders))
print("Number of images in prediction set : ", number_of_images_in_prediction_set)

plt.show()
############
train_datagen = ImageDataGenerator( rescale = 1.0/255.,shear_range=0.2,zoom_range=0.2)

# skalujemy dane 1.0/255 -  normalizacja
train_generator = train_datagen.flow_from_directory(seg_train_folders,
                                                    batch_size=32,
                                                    shuffle=True,
                                                    class_mode='categorical',
                                                    target_size=(224, 224))

validation_datagen = ImageDataGenerator(rescale = 1.0/255.) #normalizacja danych
validation_generator = validation_datagen.flow_from_directory(seg_test_folders, shuffle=True, class_mode='categorical', target_size=(224, 224))

##############
train_ds = tf.keras.utils.image_dataset_from_directory(
  seg_train_folders,
  seed=123,
  image_size=(224,224),
  batch_size=32)

##############

class_names = train_ds.class_names
print(class_names)

#Wizualizacja danych
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

#konfiguracja epok, budowa architektury sieci neuronowej, CNN
benchmark_epoch = 8

benchmark_model = Sequential()
benchmark_model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224,224,3)))
benchmark_model.add(tf.keras.layers.MaxPooling2D(2,2))
benchmark_model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
benchmark_model.add(tf.keras.layers.MaxPooling2D(2,2))
benchmark_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
benchmark_model.add(tf.keras.layers.MaxPooling2D(2,2))
benchmark_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
benchmark_model.add(tf.keras.layers.MaxPooling2D(2,2))
benchmark_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
benchmark_model.add(tf.keras.layers.MaxPooling2D(2,2))
benchmark_model.add(tf.keras.layers.Flatten())
benchmark_model.add(tf.keras.layers.Dense(1024, activation='relu'))
benchmark_model.add(tf.keras.layers.Dropout(0.2))
benchmark_model.add(tf.keras.layers.Dense(128, activation='relu'))
benchmark_model.add(tf.keras.layers.Dropout(0.2))
benchmark_model.add(tf.keras.layers.Dense(6, activation='softmax'))
benchmark_model.summary()