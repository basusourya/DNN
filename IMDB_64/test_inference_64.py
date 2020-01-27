import keras
from keras.datasets import imdb
from keras.models import load_model
import h5py
from utils_64 import utilClass as u
import numpy as np
from compressor_64 import compress as comp
from inference_64 import DNNInference as di
from keras.preprocessing.text import Tokenizer
import time
class inference(object):
	
	uc = u()

	def std_inference(self, model_name):
		
		num_classes = 10

		model = load_model(model_name)
		weights = model.get_weights()
		(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)
		tokenizer = Tokenizer(num_words=1000)
		x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
		x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
		#y_test = keras.utils.to_categorical(y_test, num_classes) # converts numerical value to array of 10x1

		w1 = np.matrix(weights[0])
		w2 = np.matrix(weights[1])
		w3 = np.matrix(weights[2])
		accuracy = 0

		for i in range(x_test.shape[0]):

			x = np.matrix(x_test[i])
			h1 = self.uc.ReLU(x*w1)
			h2 = self.uc.ReLU(h1*w2)
			o = self.uc.sigmoid(h2*w3)

			if np.argmax(o) == y_test[i]:
				accuracy = accuracy + 1

		print ('Test Result', accuracy/x_test.shape[0])
		#Test Result 0.9348


	def sorted_inference(self, model_name):

		num_classes = 10

		model = load_model(model_name)
		weights = model.get_weights()
		(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)
		tokenizer = Tokenizer(num_words=1000)
		x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
		x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
		#y_test = keras.utils.to_categorical(y_test, num_classes) # converts numerical value to array of 10x1
		
		w1 = np.matrix(weights[0])
		w2 = np.matrix(weights[1])
		w2 = self.uc.sort_weight_matrices(w1, w2)
		w3 = np.matrix(weights[2])
		w3 = self.uc.sort_weight_matrices(w2, w3)
		w1 = self.uc.sort_weight_matrix(w1)
		w2 = self.uc.sort_weight_matrix(w2)

		accuracy = 0

		for i in range(x_test.shape[0]):

			x = np.matrix(x_test[i])
			h1 = self.uc.ReLU(x*w1)
			h2 = self.uc.ReLU(h1*w2)
			o = self.uc.sigmoid(h2*w3)

			if np.argmax(o) == y_test[i]:
				accuracy = accuracy + 1

		print ('Test Result', accuracy/x_test.shape[0])
		#Test Result 0.9348


	def compressed_inference(self, model_name):

		num_classes = 10

		model = load_model(model_name)
		weights = model.get_weights()
		(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)
		tokenizer = Tokenizer(num_words=1000)
		x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
		x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
		#y_test = keras.utils.to_categorical(y_test, num_classes) # converts numerical value to array of 10x1
		
		w1 = np.matrix(weights[0])
		w2 = np.matrix(weights[1])
		w2 = self.uc.sort_weight_matrices(w1, w2)
		w3 = np.matrix(weights[2])
		w3 = self.uc.sort_weight_matrices(w2, w3)
		w1 = self.uc.sort_weight_matrix(w1)
		w2 = self.uc.sort_weight_matrix(w2)
		c = comp()
		inf = di()
		start_time = time.time()
		c.compress_network(model_name)
		end_time = time.time()
		print('Time taken for compression', end_time - start_time)
		accuracy = 0
		accuracy_comp = 0
		x_test_list = x_test.tolist()
		avg_avg_queue_length1 = 0
		avg_avg_queue_length2 = 0
		avg_max_queue_length1 = 0
		avg_max_queue_length2 = 0
		max_max_queue_length1 = 0
		max_max_queue_length2 = 0
		start_time = time.time()
		for i in range(1):
			
			print('Inferring...', i)
			x = x_test_list[i]
			x_np = np.array(x)
			h1_comp, avg_queue_length1, max_queue_length1 = inf.inferenceNN(x, w1.shape[0], w1.shape[1], c.overall_frequencies, c.comp_weight1, 'ReLU')
			h1 = self.uc.ReLU(x_np*w1)
			avg_avg_queue_length1 += avg_queue_length1
			avg_max_queue_length1 += max_queue_length1
			max_max_queue_length1 = max(max_queue_length1, max_max_queue_length1)
			#print(h1_comp,h1)
#			if(h1 != h1_comp):
#				print('Error h1', i)
#				break
			
			h2 = self.uc.ReLU(h1*w2)
			h2_comp, avg_queue_length2, max_queue_length2 = inf.inferenceNN(h1_comp, w2.shape[0], w2.shape[1], c.overall_frequencies, c.comp_weight2, 'ReLU')
			avg_avg_queue_length2 += avg_queue_length2
			avg_max_queue_length2 += max_queue_length2
			max_max_queue_length2 = max(max_queue_length2, max_max_queue_length2)
#			if(h2 != h2_comp):
#				print('Error h2', i)
#				break			
			o = self.uc.sigmoid(h2*w3)
			o_comp = self.uc.sigmoid(h2_comp*w3)
#			if(o != o_comp):
#				print('Error o_comp', i)
#				break

			if np.argmax(o) == y_test[i]:
				accuracy = accuracy + 1
			if np.argmax(o_comp) == y_test[i]:
				accuracy_comp = accuracy_comp + 1

		i = 0
		end_time = time.time()
		print('Average time taken for compressed inference', (end_time - start_time)/(i+1))
		
		start_time = time.time()
		
		i = x_np.size
		(j,k) = w1.shape
		h1_linear = [0 for l in range(k)]
		for kk in range(k):
			for ii in range(i):
				
				h1_linear[kk] = h1_linear[kk] + x_np[ii]*w1[ii,kk]

		h1_linear_np = np.array(h1_linear)
		h1 = self.uc.ReLU(h1_linear_np)

		i = h1.size
		(j,k) = w2.shape
		h2_linear = [0 for l in range(k)]
		for kk in range(k):
			for ii in range(i):
				
				h2_linear[kk] = h2_linear[kk] + h1[ii]*w2[ii,kk]

		h2_linear_np = np.array(h2_linear)
		h2 = self.uc.ReLU(h2_linear_np)

		o = self.uc.sigmoid(h2*w3)

		end_time = time.time()
		t = end_time - start_time
		print("Time without compression = ", t)

		print('avg_queue_length1', avg_avg_queue_length1/(i+1))
		print('avg_queue_length2', avg_avg_queue_length2/(i+1))
		print('avg_max_queue_length1', avg_max_queue_length1/(i+1))
		print('avg_max_queue_length2', avg_max_queue_length2/(i+1))
		print('max_queue_length1', max_queue_length1)
		print('max_queue_length2', max_queue_length2)
		print ('Test Result', accuracy/(i+1))
		print ('Test Result_comp', accuracy_comp/(i+1))			





