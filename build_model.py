import numpy as np
import pandas as pd
import random
import os
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import plot_model
from sklearn.metrics import classification_report
from collections import Counter
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import Model, layers
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Input, Conv2D, Flatten, \
    MaxPooling3D
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from keras.preprocessing import image
from tensorflow.python.keras.models import model_from_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
root_path = 'static/Images/'
train_pred_test_folders = os.listdir(root_path)
seg_train_folders = 'static/Images/seg_train/'
seg_test_folders = 'static/Images/seg_test/'
seg_pred_folders = 'static/Images/seg_pred/'
number_tra = {}
number_tes = {}
for folder in os.listdir(seg_train_folders):
    number_tra[folder] = len(os.listdir(seg_train_folders + folder))

for folder in os.listdir(seg_test_folders):
    number_tes[folder] = len(os.listdir(seg_test_folders + folder))

number_train = pd.DataFrame(list(number_tra.items()), index=range(0, len(number_tra)), columns=['class_train', 'count'])
number_test = pd.DataFrame(list(number_tes.items()), index=range(0, len(number_tes)), columns=['class_test', 'count'])

figure, ax = plt.subplots(1, 2, figsize=(20, 6))
sns.barplot(x='class_train', y='count', data=number_train, ax=ax[0])
sns.barplot(x='class_test', y='count', data=number_test, ax=ax[1])

print("Number of images in the train set : ", sum(number_tra.values()))
print("Number of images in the test set ; ", sum(number_tes.values()))
number_of_images_in_prediction_set = len(os.listdir(seg_pred_folders))
print("Number of images in prediction set : ", number_of_images_in_prediction_set)
plt.show()
############
train_datagen = ImageDataGenerator(rescale=1.0 / 255., shear_range=0.2, zoom_range=0.2)

# scale the data 1.0/255 - normalization
train_generator = train_datagen.flow_from_directory(seg_train_folders,
                                                    batch_size=32,
                                                    shuffle=True,
                                                    class_mode='categorical',
                                                    target_size=(224, 224))

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
validation_generator = validation_datagen.flow_from_directory(seg_test_folders, shuffle=True, class_mode='categorical',
                                                              target_size=(224, 224))

##############
train_ds = tf.keras.utils.image_dataset_from_directory(
    seg_train_folders,
    seed=123,
    image_size=(224, 224),
    batch_size=32)

##############
class_names = train_ds.class_names
print(class_names)

# visualization of data
plt.figure(figsize=(15, 15))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# epoch configuration, construction of neural network architecture - CNN
benchmark_epoch = 8
benchmark_model = Sequential()
benchmark_model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
benchmark_model.add(tf.keras.layers.MaxPooling2D(2, 2))
benchmark_model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
benchmark_model.add(tf.keras.layers.MaxPooling2D(2, 2))
benchmark_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
benchmark_model.add(tf.keras.layers.MaxPooling2D(2, 2))
benchmark_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
benchmark_model.add(tf.keras.layers.MaxPooling2D(2, 2))
benchmark_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
benchmark_model.add(tf.keras.layers.MaxPooling2D(2, 2))
benchmark_model.add(tf.keras.layers.Flatten())
benchmark_model.add(tf.keras.layers.Dense(1024, activation='relu'))
benchmark_model.add(tf.keras.layers.Dropout(0.2))
benchmark_model.add(tf.keras.layers.Dense(128, activation='relu'))
benchmark_model.add(tf.keras.layers.Dropout(0.2))
benchmark_model.add(tf.keras.layers.Dense(6, activation='softmax'))
print(benchmark_model.summary())
# compilation of the neural network model
benchmark_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc'])

# set EARLY STOPPING CALL  model check point API
earlystopping = EarlyStopping(monitor='val_loss',
                              patience=2,
                              verbose=1,
                              mode='min'
                              )
checkpointer = ModelCheckpoint(filepath='best_model_value', verbose=0, save_best_only=True)
callback_list = [checkpointer, earlystopping]

number_of_iterations = 14034 / 32
# training of the model
history_model = benchmark_model.fit(train_generator, epochs=benchmark_epoch, verbose=1,
                                    validation_data=validation_generator, callbacks=callback_list)

# evaluation (assessment) of the model
loss, accuracy = benchmark_model.evaluate(validation_generator)
print(f"Loss: {loss:.2f}, Accuracy: {accuracy * 100:.2f}%")

# graph of the  training i validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history_model.history['acc'], label='Training Accuracy')
plt.plot(history_model.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# graph of the training i validation loss
plt.figure(figsize=(12, 6))
plt.plot(history_model.history['loss'], label='Training Loss')
plt.plot(history_model.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


def predict_image(filename, model):
    img_ = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    plt.title(f"Prediction {str(class_names[index])}", size=20, color='blue')
    plt.imshow(img_array)


predict_image('static/Images/seg_pred/183.jpg', benchmark_model)
predict_image('static/Images/seg_pred/171.jpg', benchmark_model)
predict_image('static/Images/seg_pred/222.jpg', benchmark_model)
predict_image('static/Images/seg_pred/182.jpg', benchmark_model)
predict_image('static/Images/seg_pred/5619.jpg', benchmark_model)
predict_image('static/Images/seg_pred/5151.jpg', benchmark_model)

# saving of the model
# modelCNN_json - architecture of the neural network
# model.h5 - binary file with weights
modelCNN_json = benchmark_model.to_json()
with open("modelCNN.json", "w") as json_file:
    json_file.write(modelCNN_json)

# saving the weights to HDF5 format
benchmark_model.save_weights("model.h5")

# assignment to a variable of a previously created model
json_file = open('modelCNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# loading weights in the new model
loaded_model.load_weights("model.h5")
print("Loaded previously trained model")
