#providing utility functions like binary search and sorting algorithms.
import numpy as np
from fractions import Fraction as f
from math import log2
class utilClass(object):

	def b_search(self, array, key): #binary search

		s = len(array)

		if s == 1:
			return array[0]

		while 1:
			if key <= array[int(s/2)] and key >= array[int(s/2)-1]:
				return array[int(s/2)-1]
			elif key < array[int(s/2)-1]:
				return self.b_search(array[0:int(s/2)],key)
			elif key > array[int(s/2)]:
				return self.b_search(array[int(s/2):],key)


	def sort_weight_matrices(self, w1, w2):

		w1 = w1.transpose()
		w1 = w1.tolist()
		w2 = w2.tolist()
		w2 = [y for x,y in sorted(zip(w1,w2))]

		return np.matrix(w2)

	def sort_weight_matrix(self, w1):

		w1 = np.transpose(w1)
		w1l = w1.tolist()
		w1l = sorted(w1l)
		w1 = np.matrix(w1l)
		w1 = np.transpose(w1)

		return w1

	def index_to_weight(self, i):

		return (i-16)/100

	def weight_to_index(self, w):

		return int(round(w*100 + 16))

	def get_model_frequencies(self, weights, num_symbols):

		count = [0]*num_symbols
		total_count = 0
		for i in range(len(weights)):
			total_count += len(weights[i])*len(weights[i][0])
			for j in range(len(weights[i])):
				for k in range(len(weights[i][j])):
					count[self.weight_to_index(weights[i][j][k])] += 1

		return count

	def get_weight_frequencies(self, weight, num_symbols):

		count = [0]*num_symbols
		total_count = len(weight)*len(weight[0])
		for i in range(len(weight)):
			for j in range(len(weight[i])):
				#for k in range(len(weights[i][j])):
				count[self.weight_to_index(weight[i][j])] += 1
				#count[weight[i][j]] += 1
		return count


	def ReLU(self,x):
		return np.multiply(x,(x>0))

	def sigmoid(self,x):
	    return 1. / (1 + np.exp(-x))

	def calculate_entropy(self, freqs):
		entropy = 0
		total_freq = sum(freqs)
		p = [f(x, total_freq) for x in freqs]
		for i in range(len(p)):
			if p[i] > 0:
				entropy = entropy - p[i]*log2(p[i])

		return entropy