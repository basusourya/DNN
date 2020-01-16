from NNCompressor_64 import Node, scalarNode, Compressor
from keras.models import load_model
import numpy as np
import h5py
from utils_64 import utilClass as u
from math import log2
class compress(object):
	uc = u()
	comp_weight1 = 0
	comp_weight2 = 0
	comp_weight3 = 0
	comp_weight4 = 0
	overall_frequencies = 0
	empirical_entropy = 0
	node_w1 = 0
	node_w2 = 0
	node_w3 = 0
	node_w4 = 0
	node_w5 = 0
	def compress_network(self, model_name):
		model = load_model(model_name)
		weights = model.get_weights()
		frequencies = self.uc.get_model_frequencies(weights, 65)
		self.overall_frequencies = frequencies
		entropy = self.uc.calculate_entropy(frequencies)
		self.empirical_entropy = entropy
		w1 = np.matrix(weights[0])
		w2 = np.matrix(weights[1])
		w2 = self.uc.sort_weight_matrices(w1, w2)
		w3 = np.matrix(weights[2])
		w3 = self.uc.sort_weight_matrices(w2, w3)
		w4 = np.matrix(weights[3])
		w4 = self.uc.sort_weight_matrices(w3, w4)
		w5 = np.matrix(weights[4])
		w5 = self.uc.sort_weight_matrices(w4, w5)
		w1t = w1.transpose()
		w1l = w1t.tolist()
		w2t = w2.transpose()
		w2l = w2t.tolist()
		w3t = w3.transpose()
		w3l = w3t.tolist()
		w4t = w4.transpose()
		w4l = w4t.tolist()
		w5t = w5.transpose()
		w5l = w5t.tolist()
		print(w1.shape[0], w1.shape[1])
		print(w2.shape[0], w2.shape[1])
		print(w3.shape[0], w3.shape[1])
		print(w4.shape[0], w4.shape[1])
		print(w5.shape[0], w5.shape[1])

		comp_net = Compressor()
		
		L1 = comp_net.form_and_compress_tree(w1l, 65, frequencies)
		expected_length = (w1.shape[0]*w1.shape[1])
		print('M = ',w1.shape[0], 'N = ',w1.shape[1])
		print('Expected Length WITHOUT compression for sets = ', expected_length*entropy)
		print('MxNxH(p) - Nxlog_2(N) = ', expected_length*entropy - w1.shape[1]*log2(w1.shape[1]))
		print('Actual Length = ', len(L1))
		self.comp_weight1 = L1

		L2 = comp_net.form_and_compress_tree(w2l, 65, frequencies)
		expected_length = (w2.shape[0]*w2.shape[1])
		print('M = ',w2.shape[0], 'N = ',w2.shape[1])
		print('Expected Length WITHOUT compression for sets = ', expected_length*entropy)
		print('MxNxH(p) - Nxlog_2(N) = ', expected_length*entropy - w2.shape[1]*log2(w2.shape[1]))
		print('Actual Length = ', len(L2))
		self.comp_weight2 = L2

		L3 = comp_net.form_and_compress_tree(w3l, 65, frequencies)
		expected_length = (w3.shape[0]*w3.shape[1])
		print('M = ',w3.shape[0], 'N = ',w3.shape[1])
		print('Expected Length WITHOUT compression for sets = ', expected_length*entropy)
		print('MxNxH(p) - Nxlog_2(N) = ', expected_length*entropy - w3.shape[1]*log2(w3.shape[1]))
		print('Actual Length = ', len(L3))
		self.comp_weight3 = L3

		L4 = comp_net.form_and_compress_tree(w4l, 65, frequencies)
		expected_length = (w4.shape[0]*w4.shape[1])
		print('M = ',w4.shape[0], 'N = ',w4.shape[1])
		print('Expected Length WITHOUT compression for sets = ', expected_length*entropy)
		print('MxNxH(p) - Nxlog_2(N) = ', expected_length*entropy - w4.shape[1]*log2(w4.shape[1]))
		print('Actual Length = ', len(L4))
		self.comp_weight4 = L4
		
