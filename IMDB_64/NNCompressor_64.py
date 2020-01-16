from fractions import Fraction as f
from encoder import encoders as ec
from collections import deque
from utils_64 import utilClass as uc
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

	def form_and_compress_tree(self,w,k,overall_freqs): #Takes as input a scalarNode with 

		enc = arithmeticcoding.ArithmeticEncoder()
		newScalarNode = scalarNode(len(w), -1, 0, len(w), 0)
		q = deque([newScalarNode])

		while len(q)!=0: 
			
			temp_node = q.popleft()
			v = temp_node.v
			l = temp_node.l
			s = temp_node.s
			i = temp_node.i
			if v>0:
				temp_w = w[i:i+s]
				temp_w.sort(key = lambda x : x[l])
				w[i:i+s] = temp_w[0:s]
				count = [0 for j in range(k)]
				index = [j for j in range(k)]
				aggCount = 0

				for j in range(s):
					count[uc().weight_to_index(w[i+j][l])] += 1 #colour is from 0 to k-1

				if v>1:
					for j in range(k):
						if v>0:
							newScalarNode = scalarNode(count[j],j,i+aggCount,count[j],l+1)
							aggCount = aggCount+count[j] #constraint on l since for any given l the child nodes are being encoded
							if count[j] > 0 and l < len(w[0])-1:
								q.append(newScalarNode)
							binomial_frequencies = ec().binomial_encoder_frequencies(overall_freqs[j:], v) #can speed be improved here
							freqs = arithmeticcoding.SimpleFrequencyTable(binomial_frequencies)
							enc.write(freqs, count[j])
							v = v - count[j]
				elif v==1:
					for j in range(k):
						if count[j]==1:
							newScalarNode = scalarNode(count[j],j,i+aggCount,count[j],l+1)
							aggCount = aggCount+count[j] #constraint on l since for any given l the child nodes are being encoded
							if count[j] > 0 and l < len(w[0])-1:
								q.append(newScalarNode)
							freqs = arithmeticcoding.SimpleFrequencyTable(overall_freqs)
							enc.write(freqs, j)


		compressed_tree = enc.finish()
		return compressed_tree



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
class scalarNode(object): #The one used in BFS

	def __init__(self,val,c,i,s,l):
		self.v = val # value
		self.c = c #colour
		self.i = i #starting row index
		self.s = s #size of interval, i.e. ending index = i + s - 1
		self.l = l #column number

# inference
class Node(object): #The one used in recurssive DFS

	def __init__(self,val,k,c):
		self.childNodes = [None for i in range(k)]
		self.v = val # value
		self.c = c #colour

