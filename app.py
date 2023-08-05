# Importing required libs
from flask import Flask, render_template, request
from model import preprocess_img, predict_result
import os

IMAGES_FOLDER = os.path.join('Images', 'seg_pred')
# Instantiating flask app
app = Flask(__name__, static_folder = 'Images')
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER

# Home route
@app.route("/")
def main():
    full_filename = os.path.join(app.config['IMAGES_FOLDER'], '7110.jpg')
    img = preprocess_img(full_filename)
    pred2 = predict_result(img)

    return render_template("index.html", class_names = str(pred2), filename = full_filename)


# Prediction route
@app.route('/predict', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))


    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)