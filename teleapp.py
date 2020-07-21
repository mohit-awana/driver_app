import numpy as np
import os
from os import listdir
from os.path import isfile, join
import shutil

from helper import *

import cv2
import pandas as pd
from datetime import datetime

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

loaded_model = load_driver_distraction()
labels = ['driving safe', 'texting ', 'talking on the phone', 'operating the radio', 'drinking a beverage', 'reaching behind', 'talking to passenger']


@app.route("/")
def index():
    return render_template('index1.0.html',label='')

@app.route('/dba')
def index_pattern_analysis():
    return render_template('index_1.html')

@app.route('/data-dba', methods=['GET','POST'])
def data_overview_behaviour():
    if request.method == 'POST':
        df1 = pd.read_csv(request.files.get('file'))
        col = ['max_acc','avg_acc','avg_speed','avg_rotation','acc_roll_angle','acc_pitch_angle']
        df2 = df1[col]
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        dat = df1
        prediction = driver_preprocess_behaviour(dat)
        global var
        var = datetime.now().strftime('Driver_Behaviour_Pattern-%Y-%m-%d-%H-%M-%S.csv')
        prediction.to_csv(os.path.join(app.config['UPLOAD_FOLDER'],var), index=False)
        return render_template('data.html',data=df2.head().to_html(index=False),shape = df2.shape)
    return render_template('data.html')
    
@app.route('/dba-predict', methods=['GET','POST'])
def pattern_predict():
    path = os.path.join(app.config['UPLOAD_FOLDER'], var)
    data = pd.read_csv(path)
    return render_template('dba_result.html',data=data.to_html(index=False))

@app.route("/dist")
def distraction_page():
    return render_template('index1.1.html',label='')

@app.route("/conv")
def vid_frame():
    return render_template('index2.html',label='')

@app.route("/info", methods=['GET', 'POST'])
def vid_frame_con():
    if request.method == 'POST':
        file = request.files['file']
        if file :
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            rate = float(request.form['SaveRate'])
            frame_created, path_folder = frame(file_path,rate)
            absolute_path = "Frames saved at " + str(path_folder)
            files_in_path = [f for f in listdir(path_folder) if isfile(join(path_folder, f))]
            dir_p = absoluteFilePaths(path_folder)
            newpath = shutil.copy(dir_p[0], app.config['UPLOAD_FOLDER'])
            newpath2 = shutil.copy(dir_p[-1], app.config['UPLOAD_FOLDER'])
            newpath3 = shutil.copy(dir_p[-2], app.config['UPLOAD_FOLDER'])
        if frame_created == True:
            return render_template('index3.html', label=absolute_path, imagesrc=newpath, secondsnap=newpath2, thirdsnap=newpath3)
        else:
            return render_template('index3.html', label='FAILURE')

@app.route("/pred")
def distraction_pred_index():
    return render_template('index4.html', label='')

@app.route('/result', methods=['GET', 'POST'])
def final_result_distraction():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)    
            print("FILE PATH:", file_path)       
            preds = Predict_driver(file_path, loaded_model)
            output = "Driver is " + str(labels[int(np.argmax(preds))])
        return render_template("result.html", label=output, imagesource=file_path)

@app.route('/aps')
def aps_index():
    return render_template('index_aps1.html')

@app.route('/aps-result', methods=['GET', 'POST'])
def aps_pred():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'), header = 14, na_values = ['nan', 'na',' na'])
        cm = get_aps_pred(df)

        if cm[200] == 1:
            display = "Oops! Your vehicle needs maintenance."
            display_image = os.path.join(app.config['UPLOAD_FOLDER'], 'maintain.jpg')
        else:
            display = "Great! Your vehicle do not require maintenance."
            display_image = os.path.join(app.config['UPLOAD_FOLDER'], 'smooth.jpg')

    return render_template('result_aps.html',imagesource=display_image, label=display)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.errorhandler(UnboundLocalError)
@app.errorhandler(IndexError)
@app.errorhandler(pd.errors.ParserError)
@app.errorhandler(KeyError)
@app.errorhandler(UnicodeDecodeError)
def runtime_error(e):
    return render_template('error.html', error=str(e))

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    #app.run(host='0.0.0.0', port= port)
    app.run(debug=True)