import argparse
import base64
from datetime import datetime
import os
import shutil
import cv2

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)

cropped_height = 66
cropped_width = 200

def format_image(img):
    # crop to 66*320*3
    ret_img = img[70:70+cropped_height,:,:]
    # apply little blur
    ret_img = cv2.GaussianBlur(ret_img, (3,3), 0)
    # scale to 66x200x3
    ret_img = cv2.resize(ret_img,(200, 66), interpolation = cv2.INTER_AREA)
    # convert to YUV color space
    return  cv2.cvtColor(ret_img, cv2.COLOR_RGB2YUV)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        ori_image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(ori_image)
        X_train_data = np.zeros((1, cropped_height, cropped_width, 3))
        X_train_data[0] = format_image(image_array)

        steering_angle = float(model.predict(X_train_data, batch_size=1))

        throttle = controller.update(float(speed))

        send_control(steering_angle, throttle)
        
        base_path = './recorded_images/'
        # save frame
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        image_filename = os.path.join(base_path, timestamp)
        ori_image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':

    # check that model Keras version is same as local Keras version
    model_file = 'model.h5'
    # f = h5py.File(args.model, mode='r')
    f = h5py.File(model_file, mode='r')    
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(model_file)
    recored_images_folder = 'recorded_images'
    if recored_images_folder != '':
        print("Creating image folder at {}".format(recored_images_folder))
        if not os.path.exists(recored_images_folder):
            os.makedirs(recored_images_folder)
        else:
            shutil.rmtree(recored_images_folder)
            os.makedirs(recored_images_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's mi
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)