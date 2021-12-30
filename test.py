import tensorflow as tf
from keras.models import load_model
trainTestPath = "D:/internship/MoodMe/test/archive/images/train/angry/22.jpg"

import numpy as np
# image = tf.keras.preprocessing.image.load_img(trainTestPath,target_size = (48,48))
# image = tf.image.rgb_to_grayscale(image)
# input_arr = tf.keras.preprocessing.image.img_to_array(image)
# input_arr = np.array([input_arr])


def loadModel(trainTestPath):
    model = load_model('test/checkpoint/best_model.h5')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    tf.keras.preprocessing.image.load_img(
        trainTestPath, color_mode="grayscale", target_size=(48, 48), interpolation="nearest"
    )
    image = tf.keras.preprocessing.image.load_img(trainTestPath, target_size=(48, 48))
    image = tf.image.rgb_to_grayscale(image)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    y_pred = model.predict(input_arr)

    index = np.argmax(y_pred)
    emotion = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']
    return emotion[index]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray




