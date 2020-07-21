import os
from os.path import isfile, join
import pandas as pd
import joblib
import pickle
from keras.models import model_from_json
import cv2
import numpy as np
import itertools
from statistics import mode
from sklearn.metrics import confusion_matrix
from datetime import datetime

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
sess = tf.compat.v1.Session()

graph = tf.compat.v1.get_default_graph()

set_session(sess)


def Predict_driver(IMG_PATH,model):
    """Returns prediction for driver distraction model
    
    input : image path, model 
    
    """
    
    full_size_image = cv2.imread(IMG_PATH)
    full_size_image =cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
    full_size_image = np.expand_dims(np.expand_dims(cv2.resize(full_size_image, (240,240)), -1), 0)
    with graph.as_default():
        set_session(sess)
        result = model.predict(full_size_image)
    
    return result

def load_light_model():
    """return light gbm model"""
    filename = 'models/LightGBM.pkl'
    model_light = pickle.load(open(filename, 'rb'))
    return model_light

#model aps
def load_aps_model():
    """
    Returns aps model with feature name used for prediction in terms of list of columns
    """
    pickle_in = open("models/finalized_aps_model.pickle","rb")
    rb = pickle.load(pickle_in)
    #columns to be selected
    col = ['aa_000','ac_000','ae_000','af_000','ag_000','ag_001','ag_002','ag_003','ag_004','ag_005','ag_006','ag_007','ag_008','ag_009','ah_000',
    'ai_000','aj_000','ak_000','al_000','am_0','an_000','ao_000','ap_000','aq_000','ar_000','as_000','at_000','au_000','av_000','ax_000','ay_000',
    'ay_001','ay_002','ay_003','ay_004','ay_005','ay_006','ay_007','ay_008','ay_009','az_000','az_001','az_002','az_003','az_004','az_005','az_006',
    'az_007','az_008','az_009','ba_000','ba_001','ba_002','ba_003','ba_004','ba_005','ba_006','ba_007','ba_008','ba_009','bb_000','bc_000','bd_000',
    'be_000','bf_000','bg_000','bh_000','bi_000','bj_000','bs_000','bt_000','bu_000','bv_000','bx_000','by_000','bz_000','ca_000','cb_000','cc_000',
    'ce_000','ci_000','cj_000','ck_000','cn_000','cn_001','cn_002','cn_003','cn_004','cn_005','cn_006','cn_007','cn_008','cn_009','cp_000','cq_000',
    'cs_000','cs_001','cs_002','cs_003','cs_004','cs_005','cs_006','cs_007','cs_008','cs_009','dd_000','de_000','df_000','dg_000','dh_000','di_000',
    'dj_000','dk_000','dl_000','dm_000','dn_000','do_000','dp_000','dq_000','dr_000','ds_000','dt_000','du_000','dv_000','dx_000','dy_000','dz_000',
    'ea_000','eb_000','ee_000','ee_001','ee_002','ee_003','ee_004','ee_005','ee_006','ee_007','ee_008','ee_009','ef_000','eg_000']

    return rb,col

def load_driver_distraction():
    """Returns driver distration model """
    # load json and create model
    json_file = open('models/model_drive_activity.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model_drive_activity.h5")
    return loaded_model

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    """
    returns file with allowed extensions
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def get_download_path():
    """Returns the default downloads path for linux or windows"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'downloads')


def replacing_missing_values(df):
    """"Replaces missing values with mean, mode and median in aps model"""
    for i in df.columns:
        if df[i].nunique() < 50:
            df[i] = df[i].fillna(df[i].mode()[0])

        if df[i].nunique() > 50 and df[i].nunique() < 100:
            df[i] = df[i].fillna(df[i].mean())

        if df[i].nunique() > 100:
            df[i] = df[i].fillna(df[i].median())

def absoluteFilePaths(d):
    """get absolute path of all files in a folder"""
    all_path = []
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        if os.path.isfile(full_path):
            all_path.append(full_path)
    
    return all_path


def frame(IMG_PATH,save_rate,extension=".jpg"):
    """converts video into frames"""
    path = get_download_path()
    frames = path + "\\" + "Frames"
    new_folder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path_folder = frames + "\\" + new_folder 
    if os.path.exists(path_folder):
        pass
    else:
        os.makedirs(path_folder)
    cap = cv2.VideoCapture(IMG_PATH)
    success, frame = cap.read()
    frames_counter = 0
    img_counter = 0
    while success:
        success, frame = cap.read()
        if success:
            frames_counter+=1
            if (frames_counter % save_rate != 0):
                continue


            image_path = path_folder + "/" + str(img_counter) + extension
            img_counter+=1

            cv2.imwrite(image_path, frame)
    cap.release()
    
    if frames_counter > 0:
        return True, path_folder
    else:
        return False, path_folder


def get_aps_pred(df):
    """
    Returns labels for aps prediction model
    """
    df['class'] = pd.factorize(df['class'])[0]
    replacing_missing_values(df)
    rb,col = load_aps_model()
    x_test = df[col]
    y_test = df['class']
    y_pred_proba = rb.predict_proba(x_test)
    y_made=[]

    for i in range(0,len(y_pred_proba)):
        if y_pred_proba[i][1] >= 0.17:
            y_made.append(1)
        else:
            y_made.append(0)
    cm = confusion_matrix(y_test, y_made)
    return y_made

def driver_preprocess_behaviour(dat):
    """
    Returns prediction for driver pattern analysis
    """

    col = ['Driver_ID','max_acc','avg_acc','avg_speed','avg_rotation','acc_roll_angle','acc_pitch_angle']
    df = dat[col]  # Select data used for model creation
    model_light = load_light_model()

    #Iterate through drivers to get prediction
    for id in df.Driver_ID.unique():
        col = ['max_acc','avg_acc','avg_speed','avg_rotation','acc_roll_angle','acc_pitch_angle']
        data = df[df.Driver_ID == id]
        data = df[col]
        df['Prediction'] = model_light.predict(data)

    df['Prediction'] = df['Prediction'].map({0:'Bad',1:'Average',2:'Good'})

    df = df[['Driver_ID','Prediction']]
    #df.to_csv(datetime.now().strftime('Prediction-%Y-%m-%d-%H-%M-%S.csv'),index=False)

    # Mode of prediction column for each driver would be final prediction
    col = ['Driver_ID','Prediction']
    prediction = pd.DataFrame(columns=col) # Create DF for store predictions

    for id in df.Driver_ID.unique():
        uni = df[df.Driver_ID ==id] # dataframe for each id
        pred = uni['Prediction'].values.tolist() #convert Pred column to series
        mod = mode(pred) # take mode of list
        prediction = prediction.append({"Driver_ID" : id,"Prediction":mod},ignore_index=True) # append to dataframe
    
    return prediction