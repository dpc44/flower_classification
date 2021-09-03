
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, Response

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pickle

# new import
import urllib.request
from io import BytesIO
from PIL import Image
import cv2
import datetime, time
import shutil

#model_path = '4096-512.hdf5'
model_path = 'ResNet50_1.hdf5' #92% accuracy val
model = load_model(model_path)

a_file = open("data_array.pkl", "rb")
class_names = pickle.load(a_file)

a_file1 = open("data.pkl", "rb")
flower_dict = pickle.load(a_file1)
global capture, camera

camera = cv2.VideoCapture(1)      
app = Flask(__name__, static_folder="uploads\\")

capture=0

try:
    os.mkdir('./shots')
except OSError as error:
    pass

try:
    os.mkdir('./uploads')
except OSError as error:
    pass

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        text = request.form.get('imagefile1')
        text2 = request.form.get('imagefile2')
        imagefile = request.files['imagefile']
        #print("text2 :" , text2)
        
        count = 0
        if imagefile.filename != '':
            count = count +1
        if text != '':
            count = count +1       
        if text2 != '':
            count = count +1


        if count == 0 or count > 1:
            return redirect(request.url)

        if imagefile.filename != '':
            image_path = "./uploads/" + imagefile.filename
            
            imagefile.save(image_path)
            img = load_img(image_path, target_size=(224, 224))
        elif text != '':
            image_path = text
            res = urllib.request.urlopen(image_path).read()
            img = Image.open(BytesIO(res)).resize((224,224))
        elif text2 != '':
            shutil.copy(text2 , "./uploads/")
            basename = os.path.basename(text2)
            image_path = "./uploads/" + basename
            img = load_img(image_path, target_size=(224, 224))

        x = img_to_array(img)
        x = tf.expand_dims(x, 0)
        x = preprocess_input(x)
        preds = model.predict(x)
        preds = preds.astype('float64')
        position = np.argsort(preds[0])
        #print("------preds[0]--------")
        #print(type(preds[0]))
        #print("------List--------")
        #print(position)
        #print("------possition--------")
        #print(position[-1])
        #print(position[-2])
        #print(position[-3])
        #print("------percentage--------")
        #print(float(preds[0][position[-1]]))
        #print(float(preds[0][position[-2]]))
        #print(float(preds[0][position[-3]]))
        
        index_1 = class_names[position[-1]]
        index_2 = class_names[position[-2]]
        index_3 = class_names[position[-3]]
        int_index_1 = int(index_1)
        int_index_2 = int(index_2)
        int_index_3 = int(index_3)
        
        classification = {"class1": flower_dict[int_index_1],  "class2": flower_dict[int_index_2], "class3": flower_dict[int_index_3], "prob1": float(preds[0][position[-1]]) , "prob2":float(preds[0][position[-2]]) , "prob3":float(preds[0][position[-3]])}
        #print(image_path)
        return render_template('index.html', prediction = classification, image_path = image_path) 
    else:
        return render_template('index.html')


@app.route('/cam')
def move_camera():
    return render_template('camera.html')

@app.route('/camera', methods=['POST'])
def gen_frames():
    global capture 
    while True:
        success, frame = camera.read()  
        if not success:
            pass
        else:
            if(capture):
                capture=0
                now = datetime.datetime.now()
                name = "{}.png".format(str(now).replace(":",''))
                p = os.path.sep.join(['shots', name ])
                cv2.imwrite(p, frame)
                print("P here: ", p)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except Exception as e:
                pass

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global camera
    check = 'check'
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1    
        return render_template('camera.html', check = check)
    else:
        
        return render_template('camera.html')

@app.route('/change',methods=['POST','GET'])
def change_mode():
    global camera
    if request.method == 'POST':
        if request.form.get('submit_button') == 'Camera 1':
            camera = cv2.VideoCapture(0) 
        elif request.form.get('submit_button') == 'Camera 2':
            
            camera = cv2.VideoCapture(1)
        return render_template('camera.html')
    else:
        return render_template('camera.html')

@app.route('/Web_Image',methods=['POST','GET'])  
def send_image():
    ts = 0
    found = None
    if request.method == 'POST':
        for file_name in glob.glob('./shots/*'):
            fts = os.path.getmtime(file_name)
            if fts > ts:
                ts = fts
                found = file_name
        print("found is here: ", found)
        
        return render_template('index.html', found=found)

if __name__ == "__main__":
    app.run(debug = True)
