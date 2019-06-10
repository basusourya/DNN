import keras
from keras.datasets import mnist
from keras.models import load_model
import h5py
from utils import utilClass as u
import numpy as np
from compressor import compress as comp
from inference import DNNInference as di
import time
class inference(object):
	
	uc = u()

	def std_inference(self, model_name):
		
		num_classes = 10

		model = load_model(model_name)
		weights = model.get_weights()
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_test = x_test.reshape(10000, 784)
		x_test = x_test.astype('float32')
		x_test /= 255 
		#y_test = keras.utils.to_categorical(y_test, num_classes) # converts numerical value to array of 10x1

		w1 = np.matrix(weights[0])
		w2 = np.matrix(weights[1])
		w3 = np.matrix(weights[2])
		w4 = np.matrix(weights[3])
		w5 = np.matrix(weights[4])
		accuracy = 0

		for i in range(x_test.shape[0]):

			x = np.matrix(x_test[i])
			h1 = self.uc.ReLU(x*w1)
			h2 = self.uc.ReLU(h1*w2)
			h3 = self.uc.ReLU(h2*w3)
			h4 = self.uc.ReLU(h3*w4)
			o = self.uc.sigmoid(h4*w5)

			if np.argmax(o) == y_test[i]:
				accuracy = accuracy + 1

		print ('Test Result', accuracy/x_test.shape[0])
		#Test Result 0.9348


	def sorted_inference(self, model_name):

		num_classes = 10

		model = load_model(model_name)
		weights = model.get_weights()
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_test = x_test.reshape(10000, 784)
		x_test = x_test.astype('float32')
		x_test /= 255 
		#y_test = keras.utils.to_categorical(y_test, num_classes) # converts numerical value to array of 10x1
		
		w1 = np.matrix(weights[0])
		w2 = np.matrix(weights[1])
		w2 = self.uc.sort_weight_matrices(w1, w2)
		w3 = np.matrix(weights[2])
		w3 = self.uc.sort_weight_matrices(w2, w3)
		w4 = np.matrix(weights[3])
		w4 = self.uc.sort_weight_matrices(w3, w4)
		w5 = np.matrix(weights[4])
		w5 = self.uc.sort_weight_matrices(w4, w5)
		w1 = self.uc.sort_weight_matrix(w1)
		w2 = self.uc.sort_weight_matrix(w2)
		w3 = self.uc.sort_weight_matrix(w3)
		w4 = self.uc.sort_weight_matrix(w4)

		accuracy = 0

		for i in range(x_test.shape[0]):

			x = np.matrix(x_test[i])
			h1 = self.uc.ReLU(x*w1)
			h2 = self.uc.ReLU(h1*w2)
			h3 = self.uc.ReLU(h2*w3)
			h4 = self.uc.ReLU(h3*w4)
			o = self.uc.sigmoid(h4*w5)

			if np.argmax(o) == y_test[i]:
				accuracy = accuracy + 1

		print ('Test Result', accuracy/x_test.shape[0])
		#Test Result 0.9348


	def compressed_inference(self, model_name):

		num_classes = 10

		model = load_model(model_name)
		weights = model.get_weights()
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_test = x_test.reshape(10000, 784)
		x_test = x_test.astype('float32')
		x_test /= 255 
		#y_test = keras.utils.to_categorical(y_test, num_classes) # converts numerical value to array of 10x1
		
		w1 = np.matrix(weights[0])
		w2 = np.matrix(weights[1])
		w2 = self.uc.sort_weight_matrices(w1, w2)
		w3 = np.matrix(weights[2])
		w3 = self.uc.sort_weight_matrices(w2, w3)
		w4 = np.matrix(weights[3])
		w4 = self.uc.sort_weight_matrices(w3, w4)
		w5 = np.matrix(weights[4])
		w5 = self.uc.sort_weight_matrices(w4, w5)
		w1 = self.uc.sort_weight_matrix(w1)
		w2 = self.uc.sort_weight_matrix(w2)
		w3 = self.uc.sort_weight_matrix(w3)
		w4 = self.uc.sort_weight_matrix(w4)
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
		avg_avg_queue_length3 = 0
		avg_avg_queue_length4 = 0
		avg_max_queue_length1 = 0
		avg_max_queue_length2 = 0
		avg_max_queue_length3 = 0
		avg_max_queue_length4 = 0
		max_max_queue_length1 = 0
		max_max_queue_length2 = 0
		max_max_queue_length3 = 0
		max_max_queue_length4 = 0
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
			h3 = self.uc.ReLU(h2*w3)
			h3_comp, avg_queue_length3, max_queue_length3 = inf.inferenceNN(h2_comp, w3.shape[0], w3.shape[1], c.overall_frequencies, c.comp_weight3, 'ReLU')
			avg_avg_queue_length3 += avg_queue_length3
			avg_max_queue_length3 += max_queue_length3
			max_max_queue_length3 = max(max_queue_length3, max_max_queue_length3)
#			if(h3 != h3_comp):
#				print('Error h3', i)
#				break			
			h4 = self.uc.ReLU(h3*w4)
			h4_comp, avg_queue_length4, max_queue_length4 = inf.inferenceNN(h3_comp, w4.shape[0], w4.shape[1], c.overall_frequencies, c.comp_weight4, 'ReLU')
			avg_avg_queue_length4 += avg_queue_length4
			avg_max_queue_length4 += max_queue_length4
			max_max_queue_length4 = max(max_queue_length4, max_max_queue_length4)
#			if(h4 != h4_comp):
#				print('Error h4', i)
#				break
			o = self.uc.sigmoid(h4*w5)
			o_comp = self.uc.sigmoid(h4_comp*w5)
#			if(o != o_comp):
#				print('Error o_comp', i)
#				break

			if np.argmax(o) == y_test[i]:
				accuracy = accuracy + 1
			if np.argmax(o_comp) == y_test[i]:
				accuracy_comp = accuracy_comp + 1

		i = 0
		end_time = time.time()
		print('Average time taken for Inference', (end_time - start_time)/(i+1))

		print('avg_queue_length1', avg_avg_queue_length1/(i+1))
		print('avg_queue_length2', avg_avg_queue_length2/(i+1))
		print('avg_queue_length3', avg_avg_queue_length3/(i+1))
		print('avg_queue_length4', avg_avg_queue_length4/(i+1))
		print('avg_max_queue_length1', avg_max_queue_length1/(i+1))
		print('avg_max_queue_length2', avg_max_queue_length2/(i+1))
		print('avg_max_queue_length3', avg_max_queue_length3/(i+1))
		print('avg_max_queue_length4', avg_max_queue_length4/(i+1))
		print('max_queue_length1', max_queue_length1)
		print('max_queue_length2', max_queue_length2)
		print('max_queue_length3', max_queue_length3)
		print('max_queue_length4', max_queue_length4)
		print ('Test Result', accuracy/(i+1))
		print ('Test Result_comp', accuracy_comp/(i+1))			





