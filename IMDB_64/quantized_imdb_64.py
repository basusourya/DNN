import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from quantize import main as m


# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

# One-hot encoding the output
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(x_train.shape[1])
# Neural Network model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],), use_bias = False))
model.add(Dropout(.5))
model.add(Dense(64, activation='relu', use_bias = False))
model.add(Dropout(.5))
model.add(Dense(2, activation='sigmoid', use_bias = False))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=2, batch_size=100, verbose=1, validation_split=0.2)

# Evaluate the model
score = model.evaluate(x_train, y_train)
print("\n Training Accuracy:", score[1])
score = model.evaluate(x_test, y_test)
print("\n Testing Accuracy:", score[1])

weights = model.get_weights()
quant = m()

steps = []
		
for i in range(65):
	steps.append((i-32)/200)


for i in range(3):
	weights[i] = quant.fQ(weights[i], steps)

model.set_weights(weights)
score = model.evaluate(x_test, y_test, verbose=0)
model.save('quantized_model_imdb_64.h5')
print('Quantized test loss:', score[0])
print('Quantized test accuracy:', score[1])
