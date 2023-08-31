# Images Recognition with various nature scenes (Classification based on 6 classes, categories of images) + web application API FLASK
This web application was created using the Flask framework and is about predicting images from 
various nature scenes (classification into 1 out of 6 image categories) based on a deep learning model trained using convulsive neural networks.
Dataset (24 335 images)

Steps are:

<b>1. Run file build_model.py:</b>
+ Build the model.
+ Compile the model.
+ Train / fit the data to the model.
+ Evaluate the model on the testing set.
+ Conv2D: (32 filters of size 3 by 3) The features will be "extracted" from the image.
+ MaxPooling2D: The images get half sized.
+ Flatten: Transforms the format of the images from a 2d-array to a 1d-array of 150 150 3 pixel values.
+ Relu : given a value x, returns max(x, 0).
+ Softmax: 6 neurons, probability that the image belongs to one of the classes.
+ Save of the model (architecture, weights).

<b>2. Run file app.py - instance of the Flask API (application web - html)</b>
+ creation of two endpoints ("/", /prediction) 
+ use of Jinja2 templates, html, CSS

<b>Sets: training, test - number of classes (categories) of images</b>
<img src ="/static/Evaluate/class_train_test.png">
<b>Sample pictures</b>
<img src ="/static/Evaluate/examples_images.png">
<b>Training results</b>
<img src ="/static/Evaluate/Training_Validation_Accuracy.png">
<img src ="/static/Evaluate/Training_Validation_Loss.png">