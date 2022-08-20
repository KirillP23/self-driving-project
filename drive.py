import argparse
import base64
from datetime import datetime
import os
import shutil

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

import numpy as np
from PIL import Image
from io import BytesIO

import torch

from model import Net
from pid import SimplePIController


sio = socketio.Server()
app = Flask(__name__)

controller = SimplePIController(Kp=0.1, Ki=0.002)
set_speed = 10
controller.set_desired(set_speed)

steering_angle_alpha = 0.05

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = data["steering_angle"] # current steering angle of the car
        throttle = data["throttle"] # current throttle of the car
        speed = data["speed"] # current speed of the car
        imgString = data["image"] # current image from the center camera of the car
        imgPIL = Image.open(BytesIO(base64.b64decode(imgString)))
        img = torch.from_numpy(np.asarray(imgPIL)).float().permute(2,0,1).unsqueeze(0)
        steering_angle = float(model(img)) * steering_angle_alpha
        throttle = controller.update(float(speed))
        print(f"{steering_angle:.4f} {throttle:.4f}")
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            imgPIL.save('{}.jpg'.format(image_filename))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to model.pt file.')
    parser.add_argument('image_folder', type=str, nargs='?', default='', help='Path to image folder. This is where the images from the run will be saved.')
    args = parser.parse_args()

    model = Net()
    # Use this if you dont have GPUs
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load(args.model))
    model.eval()

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)