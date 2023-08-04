# Images Recognition with various nature scenes (Classification based on 6 classes, categories of images) + web application API FLASK
This web application was created using the Flask framework and is about predicting images from 
various nature scenes (classification into 1 out of 6 image categories) based on a deep learning model trained using convulsive neural networks.
Dataset (24 335 images)
Steps are:
+ Build the model,
+ Compile the model,
+ Train / fit the data to the model,
+ Evaluate the model on the testing set,
+ Conv2D: (32 filters of size 3 by 3) The features will be "extracted" from the image.
+ MaxPooling2D: The images get half sized.
+ Flatten: Transforms the format of the images from a 2d-array to a 1d-array of 150 150 3 pixel values.
+ Relu : given a value x, returns max(x, 0).
+ Softmax: 6 neurons, probability that the image belongs to one of the classes.
