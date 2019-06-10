from NNCompressor_64 import Node
from NNCompressor_64 import Compressor
from keras.models import load_model
import numpy as np
import h5py
from utils_64 import utilClass as u
from math import log2
from os import sys
class compress(object):
	uc = u()
	comp_weight1 = 0
	comp_weight2 = 0
	overall_frequencies = 0
	empirical_entropy = 0
	node_w1 = 0
	node_w2 = 0
	node_w3 = 0
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
		w1t = w1.transpose()
		w1l = w1t.tolist()
		w2t = w2.transpose()
		w2l = w2t.tolist()
		w3t = w3.transpose()
		w3l = w3t.tolist()
		print(w1.shape[0], w1.shape[1])
		print(w2.shape[0], w2.shape[1])
		print(w3.shape[0], w3.shape[1])
		self.node_w1 = Node(w1.shape[1],65,-1)
		self.node_w2 = Node(w2.shape[1],65,-1)
		self.node_w3 = Node(w3.shape[1],65,-1)
		comp_net = Compressor()
		sys.setrecursionlimit(1500)
		comp_net.formTree(self.node_w1,w1l,0,65)
		comp_net.formTree(self.node_w2,w2l,0,65)
		L1 = comp_net.compressTree(self.node_w1, frequencies, w1.shape[1])
		expected_length = (w1.shape[0]*w1.shape[1])
		print('M = ',w1.shape[0], 'N = ',w1.shape[1])
		print('Expected Length WITHOUT compression for sets = ', expected_length*entropy)
		print('MxNxH(p) - Nxlog_2(N) = ', expected_length*entropy - w1.shape[1]*log2(w1.shape[1]))
		print('Actual Length = ', len(L1))
		self.comp_weight1 = L1

		L2 = comp_net.compressTree(self.node_w2, frequencies, w2.shape[1])
		expected_length = (w2.shape[0]*w2.shape[1])
		print('M = ',w2.shape[0], 'N = ',w2.shape[1])
		print('Expected Length WITHOUT compression for sets = ', expected_length*entropy)
		print('MxNxH(p) - Nxlog_2(N) = ', expected_length*entropy - w2.shape[1]*log2(w2.shape[1]))
		print('Actual Length = ', len(L2))
		self.comp_weight2 = L2
