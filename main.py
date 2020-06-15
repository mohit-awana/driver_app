import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import pandas as pd
from keras.models import load_model
from keras.models import model_from_json



ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (240, 240)
UPLOAD_FOLDER = 'uploads'


    
json_file = open('model_drive_activity.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_drive_activity.h5")
print(" * Model is Loaded!")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
    # load an image file
    image = load_img(file, target_size=(240, 240))
	# convert to a numpy array
    image = img_to_array(image)
    image = cv2.resize(image,(240,240)).reshape(-1, 240, 240, 1)
	# predict probability of class
    classes = ['c0','c1_c3','c2_c4','c5','c6','c7','c9']
    classes_d = {'c0': 'safe driving',    'c1_c3': 'texting',  'c2_c4': 'talking on the phone', 'c5': 'operating the radio',    'c6': 'drinking',    'c7': 'reaching behind', 'c9': 'talking to passenger'}
    pred = model.predict_classes(image)
	# class labels
    result = classes_d[classes[pd.to_numeric(format(np.argmax(predict)))]]
    return result

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('index_1.html', label='')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
        return render_template("result.html", label=output, imagesource=file_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
