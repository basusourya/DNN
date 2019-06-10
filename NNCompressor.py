from fractions import Fraction as f
from encoder import encoders as ec
from collections import deque
from utils import utilClass as uc
import arithmeticcoding

class Compressor(object):
	j = 0

	def formTree(self,node,w,l,k):#l is the column number
		
		w.sort(key = lambda x : x[l])
		count = [0 for i in range(k)]
		aggCount = 0
		index = [i for i in range(k)]
		for i in range(len(w)):
			count[uc().weight_to_index(w[i][l])] += 1 #colour is from 0 to k-1
			#count[w[i][l]] += 1
		#count_index = zip(count, index)
		#print (list(count_index))
		
		for i in range(k):
			#self.j = self.j + 1
			#print('Forming Tree...',self.j)
			newNode = Node(count[i],k,i)
			node.childNodes[i] = newNode
			if count[i] > 0 and l < len(w[0])-1:
				self.formTree(newNode, w[aggCount:aggCount + count[i]], l+1,k )
			aggCount += count[i]

	def get_weight_matrix(self, node):

		w = [0 for i in range(10)]
		return w

	def compressTree(self, node, overall_freqs, N): #n is the number of nodes in the hidden layer and pw is the list of all the normalized probability; use cummulative frequencies, then, 
	#won't have to normalize
		enc = arithmeticcoding.ArithmeticEncoder()
		q = deque([node])
		#self.j = 0
		while len(q)!=0:
			temp = q.popleft()
			if temp.v>1:
				tempValue = temp.v   
				i = 0
				for child in temp.childNodes:
					if child != None:
						
						if tempValue > 0:
							q.append(child)
							binomial_frequencies = ec().binomial_encoder_frequencies(overall_freqs[i:], tempValue) # binomial encoder can convert to frequencies. convert to binary independently and check compression ratio for confirming correct amount of compression
							freqs = arithmeticcoding.SimpleFrequencyTable(binomial_frequencies)
							enc.write(freqs, child.v)
							tempValue = tempValue - child.v
						#a = a + '1011'
							i += 1
							#print('Compressing Tree...',self.j)
							#self.j += 1
						#print (i)
			elif temp.v == 1:
				for child in temp.childNodes:
					if child != None:
						if child.v == 1:
							symbol = child.c
							q.append(child)
							freqs = arithmeticcoding.SimpleFrequencyTable(overall_freqs)
							enc.write(freqs, symbol)



		compressed_tree = enc.finish()

		return compressed_tree

# inference
class Node(object):

	def __init__(self,val,k,c):
		self.childNodes = [None for i in range(k)]
		self.v = val # value
		self.c = c #colour

