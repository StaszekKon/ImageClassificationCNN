# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image
import tensorflow as tf
# from tensorflow.python.keras.models import model_from_json
from keras.models import model_from_json
from keras.preprocessing import image

# załądowanie architektury modelu
json_file = open('modelCNN.json','r')
loaded_model_json = json_file.read()
json_file.close()


loaded_model = model_from_json(loaded_model_json)

# załadowanie wag w nowym modelu

loaded_model.load_weights("model.h5")
# przygotowanie i preprocessing obrazu
# img_path = 'Images/seg_pred/182.jpg'
def preprocess_img(img_path):
    # op_img = Image.open(img_path)
    # img_resize = op_img.resize((224, 224))
    # img2arr = img_to_array(img_resize) / 255.0
    # img_reshape = img2arr.reshape(1, 224, 224, 3)
    # return img_reshape
    # img_path = 'Images/seg_pred/182.jpg'
    img_ = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    return img_processed

# Predicting function
def predict_result(img_processed):
    pred = loaded_model.predict(img_processed)
    # return np.argmax(pred[0], axis=-1)
    index = np.argmax(pred[0])
    if index == 0:
        class_names = "buildings"
    elif index == 1:
        class_names = "forest"
    elif index == 2:
        class_names = "glacier"
    elif index == 3:
        class_names = "mountain"
    elif index == 4:
        class_names ="sea"
    elif index == 5:
        class_names ="street"
    else:
        class_names ="zła predykcja"

    # return np.argmax(pred[0])
    return class_names

