
from utlis import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
##Step 1:
path = 'dataset'
data = importDataInfo(path)

#Step 2:
data = balanceData(data,display=False)

#Step 3:
imagesPath , steering = loadData(path,data)
#print(imagesPath[0],steering[0])

#Step 4:
xTrain,xVal,yTrain,yVal = train_test_split(imagesPath,steering,test_size=0.2,random_state=5)
#print('total training images:', len(xTrain))
#print('total training images:', len(xVal))

#Step 5:

#Step 6:
model=Model()
model.summary()

#step 9:

history = model.fit(batchGen(xTrain,yTrain,100,1),
          steps_per_epoch=300,
          epochs=10,
          validation_data = batchGen(xVal,yVal,100,0),
          validation_steps=200)


#Step 10:
model.save('training_model_0.00001.h5')
print('Saving model')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
#plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
