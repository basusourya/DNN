from fractions import Fraction as f
from NNCompressor_32 import Node
from decoder import decoders as dc 
from encoder import encoders as ec
from utils_32 import utilClass as uc
from collections import deque
import arithmeticcoding
import numpy as np
from NNCompressor_32 import Compressor as c
from math import log2, ceil, floor


class DNNInference(object):
	w = 0
	def inferenceNN(self, x, M, N, overall_freqs, L, activationFunction): #N is the number of hidden nodes, the weights are of dimension MxN
		y = [0 for i in range(N)]
		enc = arithmeticcoding.ArithmeticEncoder()
		dec = arithmeticcoding.ArithmeticDecoder(L)
		q = deque([N])
		#q_node = deque([node])
		self.w = 0
		tot_queue_length = floor(2*log2(N+1)+1)
		max_queue_length = floor(2*log2(N+1)+1)
		current_queue_length = floor(2*log2(N+1)+1)
		j = 0
		level = 0
		flag = 0
		flagp = 0
		k = len(overall_freqs)
		print('M:',M,'N:',N)
		while len(q)!=0 and level < M:
			currentNodeValue = q.popleft()
			current_queue_length -= floor(2*log2(currentNodeValue+1)+1)

			if flagp == 0:
				print('current_queue_length',current_queue_length)
				flagp = 1
			#currentnode = q_node.popleft()
			if currentNodeValue > 1:
				c = 0     						#colour initialized with 0
				while c <= k-1 and currentNodeValue > 0: #kth colour need not be encoded
					binomial_frequencies = ec().binomial_encoder_frequencies(overall_freqs[c:], currentNodeValue)
					freqs = arithmeticcoding.SimpleFrequencyTable(binomial_frequencies)
					childNodeValue = dec.read(freqs)
					#if childNodeValue != currentnode.childNodes[c].v:
					#	print('Not Matching!', childNodeValue, currentnode.childNodes[c].v)
					#else:
					#	print('No problems here')
					enc.write(freqs, childNodeValue)
					currentNodeValue -= childNodeValue
					q.append(childNodeValue)
					current_queue_length += floor(2*log2(childNodeValue+1)+1)
					max_queue_length = max(max_queue_length, current_queue_length)
					tot_queue_length += current_queue_length
					self.w += 1
					#q_node.append(currentnode.childNodes[c])
					#print('childNodeValue',childNodeValue)
					if childNodeValue > 0:
						flag = 1
					for i in range(childNodeValue):
						
					#	print('level:',level,'x[level]',x[level])
					#	print('Calculating Y....', level,':',self.w)
						y[j+i] += uc().index_to_weight(c)*x[level]
						#print(x[level], c)
						#y[j+i] += c*x[level]
					c = c+1
					j = (j + childNodeValue)%N
					if j == 0 and flag:
						level = level + 1
						#print('level:',level)
						flag = 0
			elif currentNodeValue == 1:
				freqs = arithmeticcoding.SimpleFrequencyTable(overall_freqs)
				c = dec.read(freqs)
				enc.write(freqs, c)
				q.append(1)
				current_queue_length += 3
				max_queue_length = max(max_queue_length, current_queue_length)
				tot_queue_length += current_queue_length
				self.w += 1
				i = 0
				y[j+i] += uc().index_to_weight(c)*x[level]
				j = (j + 1)%N
				if j == 0:
					level += 1

		avg_queue_length = tot_queue_length/self.w

		L1 = enc.finish() #return L1 if needed
		y = np.array(y)
		if activationFunction == 'ReLU':
			y = uc().ReLU(y)
		elif activationFunction == 'sigmoid':
			y = uc().sigmoid(y)
		elif activationFunction == None:
			y = y
		return y, avg_queue_length, max_queue_length

	def test_inference(self, x, w, num_symbols):
		'''
		x = 1x2
		w = 2x5
		y = 1x5

		'''
		
		w = w.transpose()
		print(w.shape[0], w.shape[1])
		w = w.tolist()
		node = Node(len(w), num_symbols, -1)
		c().formTree(node, w, 0, num_symbols)
		frequencies = uc().get_weight_frequencies(w, num_symbols)
		L = c().compressTree(node, frequencies, len(w))
		w1 = np.matrix(w)
		y_comp = self.inferenceNN(node, x, w1.shape[1], w1.shape[0], frequencies, L, None)
		w1 = w1.transpose()
		w1 = uc().sort_weight_matrix(w1)
		y = x*w1
		return node, y, y_comp





	def colourToWeight(self,c):
		return c-3  #convert to weight convention

	def activationFunction(self,y):
		
		ans = np.exp(-0.01*y)
		return 1/(1+ans) # use the activation function


