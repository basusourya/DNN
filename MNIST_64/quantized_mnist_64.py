from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import h5py
from quantize import main as m

batch_size = 128
num_classes = 10
epochs = 5

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(784,), use_bias = False))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu', use_bias = False))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu', use_bias = False))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu', use_bias = False))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax', use_bias = False))

model.summary()
 
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
weights = model.get_weights()
quant = m()

steps = []
		
for i in range(65):
	steps.append((i-32)/200)


for i in range(5):
	weights[i] = quant.fQ(weights[i], steps)

model.set_weights(weights)
score = model.evaluate(x_test, y_test, verbose=0)
model.save('quantized_model_64.h5')
print('Quantized test loss:', score[0])
print('Quantized test accuracy:', score[1])

'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
'''
Test loss: 0.0753689441902
Test accuracy: 0.98
Quantized test loss: 0.163888313049
Quantized test accuracy: 0.9752

'''
