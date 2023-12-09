import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


def getName(filepath):
    return filepath.split('\\')[-1]

def importDataInfo(path):
    coloums = ['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data  = pd.read_csv(os.path.join(path,'driving_log.csv'),names=coloums)
    data['Center'] = data['Center'].apply(getName)
    #print(data.head())
    print('Total images:', data.shape[0])
    return data

def balanceData(data,display=True):
    nBins = 31
    samplesBin = 500
    hist,bins = np.histogram(data['Steering'],nBins)

    if display:
         center = (bins[:-1]+bins[1:])*0.5
         plt.bar(center,hist,width=0.06)
         plt.plot((np.min(data['Steering']),np.max(data['Steering'])),(samplesBin,samplesBin))
         plt.show()

    removeIndexList = []
    for j in range(nBins):
        bindataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                bindataList.append(i)
        bindataList = shuffle(bindataList)
        bindataList = bindataList[samplesBin:]
        removeIndexList.extend(bindataList)

    print('Remove images:',len(removeIndexList) )
    data.drop(data.index[removeIndexList],inplace=True)
    print('remain images:', len(data))

    if display:
         hist, _ = np.histogram(data['Steering'],(nBins))
         plt.bar(center,hist,width=0.06)
         plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesBin, samplesBin))
         plt.show()
    return data

def loadData(path,data):
    imgPath = []
    steering = []
    for i in range(len(data)):
        indexData = data.iloc[i]
        #print(indexData)
        imgPath.append(f'{path}/IMG/{indexData[0]}')
        #print(imgPath)
        steering.append((float(indexData[3])))
    imgPath=np.asarray(imgPath)
    steering = np.asarray(steering)
    return imgPath, steering

def augmentImage(imagePath,steering):
    img = mpimg.imread(imagePath)
    #PAN
    if np.random.rand() <0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    #ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    #Brightness
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4,1.2))
        img = brightness.augment_image(img)

    #flip
    if np.random.rand() <0.5:
        img = cv2.flip(img,1)
        steering = -steering

    return img, steering

def preProcessing(img):
    img = img[60:135,:,:] #crop image
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66)) #NDiVIA model using 200x66 image
    img = img/255 #normalization, convert to 0 and 1
    return img

# imgteset= preProcessing(mpimg.imread('test.jpg'))
# plt.imshow(imgteset)
# plt.show()
def batchGen(imagePath, steeringList, batchSize,trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0,len(imagePath)-1)
            if trainFlag:
                img , steering = augmentImage(imagePath[index],steeringList[index])
            else:
                img = mpimg.imread(imagePath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield(np.asarray(imgBatch),np.asarray(steeringBatch))

def Model():
    model = Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape = (66,200,3),activation = 'elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation = 'elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation = 'elu'))
    model.add(Convolution2D(64,(3,3),activation = 'elu'))
    model.add(Convolution2D(64,(3,3),activation = 'elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1, activation='elu'))

    model.compile(Adam(learning_rate=0.00001), loss='mse')
    #model.compile(Adam(learning_rate= 0.0001),loss='mse')
    #model.compile(Adam(learning_rate=0.001), loss='mse')
    #model.compile(Adam(learning_rate=0.01), loss='mse')
    return model
