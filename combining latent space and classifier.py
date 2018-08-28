import os
import keras
from keras.utils import np_utils
from keras.layers import *
from keras.models import Model  
from keras.datasets import mnist  
import numpy as np  
import matplotlib.pyplot as plt 
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

(x_train,y_train), (x_test,y_test) = mnist.load_data() # load the data

x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)  #change the images shape from 28*28 to 784*1


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10) 
input_img = Input(shape=(784,))
hidden_1 = Dense(250, activation='relu')(input_img)
hidden_2 = Dense(225, activation='relu')(hidden_1)
classifier1 = Dense(100, activation='softmax', name='classifier1')(hidden_2) #output of function f1


rms = RMSprop()


latent_space = Dense(100, activation='sigmoid')(hidden_2) # output of function f2

merge_input = concatenate([classifier1,latent_space],axis=-1)
y = Dense(100, activation='relu')(merge_input)
y = Dense(150, activation='relu')(y)
y = Dense(250, activation='relu')(y)
main_output = Dense(784, activation='sigmoid', name='main_output')(y)

total_model = Model(input=input_img, outputs=[classifier1,main_output]) #build the model

total_model.compile(optimizer='rmsprop', loss=['mse','mse'], metrics=['accuracy'], loss_weights=[1,1])

history = total_model.fit(x_train, [y_train, x_train], epochs=200,batch_size=256,shuffle=True,validation_split=0.1) #train the model

total_model.summary()

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Total loss of whole model')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Test'],loc='upper left')
plt.show()

plt.plot(history.history['classifier1_acc'])
plt.plot(history.history['val_classifier1_acc'])  
plt.title('Classification accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show() # get the loss and accuracy plots


[result_1, result_2]= total_model.predict(x_test) #test the model



BigIm1 = np.zeros((280, 280))
index = 0
for i in range(10):
    for j in range(10):
        temp1 = result_2[index].reshape(28, 28)
        BigIm1[28*i:28*(i+1), 28*j:28*(j+1)] = temp1
        index += 1

plt.imshow(BigIm1)
plt.show()

BigIm1 = np.zeros((280, 280))
index = 0
for i in range(10):
    for j in range(10):
        temp1 = result_2[index].reshape(28, 28)
        BigIm1[28*i:28*(i+1), 28*j:28*(j+1)] = temp1
        index += 1

plt.imshow(BigIm1)
plt.gray()
plt.show()  #get the images of the test results


