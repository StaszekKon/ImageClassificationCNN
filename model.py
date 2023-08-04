# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image
from tensorflow.python.keras.models import model_from_json

# załądowanie architektury modelu
json_file = open('modelCNN.json','r')
loaded_model_json = json_file.read()
json_file.close()


loaded_model = model_from_json(loaded_model_json)

# załadowanie wag w nowym modelu

loaded_model.load_weights("model.h5")
# przygotowanie i preprocessing obrazu
def preprocess_img(img_path):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


# Predicting function
def predict_result(predict):
    pred = loaded_model.predict(predict)
    return np.argmax(pred[0], axis=-1)