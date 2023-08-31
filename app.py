from flask import Flask, render_template, request
from model import preprocess_img, predict_result
import os
from werkzeug.utils import secure_filename

IMAGES_FOLDER = os.path.join('static/Images', 'seg_pred')
# Flask app instance
app = Flask(__name__)
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER


# main endpoint
@app.route("/")
def main():
    full_filename = os.path.join(app.config['IMAGES_FOLDER'], '7110.jpg')
    img = preprocess_img(full_filename)
    recognition = predict_result(img)
    return render_template("index.html", class_names=str(recognition), filename=full_filename)


# prediction route (endpoint)
@app.route('/predict', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            uploaded_img = request.files['uploaded-file']
            img_filename = secure_filename(uploaded_img.filename)
            uploaded_img.save(os.path.join(app.config['IMAGES_FOLDER'], img_filename))
            full_filename = os.path.join(app.config['IMAGES_FOLDER'], img_filename)
            img = preprocess_img(full_filename)
            pred = predict_result(img)
            return render_template("result.html", predictions=pred, file_name_img=full_filename)

    except:
        error = "The file cannot be processed."
        return render_template("result.html", err=error)


if __name__ == "__main__":
    app.run(port=9000, debug=True)