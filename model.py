import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# loading the model architecture
json_file = open('modelCNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# loading weights in a new model
loaded_model.load_weights("model.h5")


# image preparation and preprocessing


def preprocess_img(img_path):
    img_ = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    return img_processed


# prediction
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
        class_names = "sea"
    elif index == 5:
        class_names = "street"
    else:
        class_names = "zła predykcja"

    return class_names
