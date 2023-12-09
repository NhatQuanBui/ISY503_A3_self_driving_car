import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64

from io import BytesIO
from PIL import Image
import cv2

#### FOR REAL TIME COMMUNICATION BETWEEN CLIENT AND SERVER
sio = socketio.Server()
#### FLASK IS A MICRO WEB FRAMEWORK WRITTEN IN PYTHON
app = Flask(__name__)

maxSpeed =10


def preProcessing(img):
    img = img[60:135,:,:] #crop image
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66)) #NDiVIA model using 200x66 image
    img = img/255 #normalization, convert to 0 and 1
    return img

@sio.on('telemetry')
def telemetry(sid,data):
    speed = float(data['speed'])
    image = Image.open((BytesIO(base64.b64decode(data['image']))))
    image = np.asarray(image)
    image = preProcessing(image)
    image = np.asarray([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print('{} {} {}'.format(steering,throttle,speed))
    sendControl(steering,throttle)

@sio.on('connect')
def connect(sid,environ):
    print('Connected')
    sendControl(0,0)

def sendControl(steering,throttle):
    sio.emit('steer',data={
        'steering_angle': steering.__str__(),
        'throttle' : throttle.__str__()
    })

if __name__ == '__main__':
    model = load_model('training_model_0.00001.h5')
    app = socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)
