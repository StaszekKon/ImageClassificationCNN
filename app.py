# Importing required libs
from flask import Flask, render_template, request
from model import preprocess_img, predict_result
import os
from werkzeug.utils import secure_filename

IMAGES_FOLDER = os.path.join('Images', 'seg_pred')
# Instantiating flask app
app = Flask(__name__, static_folder = 'Images')
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER

# Home route
@app.route("/")
def main():
    full_filename = os.path.join(app.config['IMAGES_FOLDER'], '7110.jpg')
    img = preprocess_img(full_filename)
    recognition = predict_result(img)

    return render_template("index.html", class_names = str(recognition), filename = full_filename)


# Prediction route
@app.route('/predict', methods=['GET'])
def predict_image_file():
    try:
        if request.method == 'GET':
            # img = preprocess_img(request.files['file'].stream)
            # full_filename = os.path.join(app.config['IMAGES_FOLDER'], '7322.jpg')
            uploaded_img = request.files['uploaded-file']
            img_filename = secure_filename(uploaded_img.filename)
            # uploaded_img.save(os.path.join(app.config['IMAGES_FOLDER'], img_filename))
            # img = preprocess_img(img_filename)
            # pred = predict_result(img)
            testS = "Ala"
            return render_template("result.html", predictions="pred", name2 = testS, img2 = img_filename)


    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)