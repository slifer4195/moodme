from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from numpy import imag
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image  import img_to_array
import tensorflow as tf 
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
from PIL import Image

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
RESIZED_FOLDER = 'static/resize/'
 
model = load_model('./best_model.h5')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESIZED_FOLDER'] = RESIZED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        im = Image.open(image_path)
        im2 = im.resize((400,400))
        im2.save(os.path.join(app.config['RESIZED_FOLDER'], filename))
        print(im2)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        tf.keras.preprocessing.image.load_img(
            image_path, color_mode="grayscale", target_size=(48, 48), interpolation="nearest"
        )
        tf.keras.preprocessing.image.load_img( image_path, color_mode="grayscale", target_size=(48, 48), interpolation="nearest")
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(48, 48))
        image = tf.image.rgb_to_grayscale(image)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        y_pred = model.predict(input_arr)

        index = np.argmax(y_pred)
        emotion = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']
        classification = emotion[index]
        flash('Image successfully uploaded and displayed below: ' + classification)

        return render_template('index.html', filename=filename, classification=classification)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='resize/' + filename), code=301)
 


if __name__ == "__main__":
    app.run()