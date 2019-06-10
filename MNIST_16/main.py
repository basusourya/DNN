from mnist import MNIST
from inference import DNNInference
from NNCompressor import Compressor, Node
import numpy as np
from fractions import Fraction as f
import time
class mainClass(object):

	def train(self):
		'''
		# Saving the objects:
			with open('objs.pickle', 'w') as f:  # Python 3: open(..., 'wb')
    			pickle.dump([obj0, obj1, obj2], f)

		# Getting back the objects:
			with open('objs.pickle') as f:  # Python 3: open(..., 'rb')
    			obj0, obj1, obj2 = pickle.load(f)
		'''
		start = time.time()
		mndata = MNIST('mnist_data')
		x_train,y_train = mndata.load_training()	#X = 60000x784; Y = 60000x1	

		alpha = 0.1
		beta = 0.01
		N = len(x_train) #number of training samples
		D = len(x_train[0])
		n_hidden = 300
		k = 10
		w1 = 2*np.random.rand(D + 1,n_hidden)-1  #785x300
		w2 = 2*np.random.rand(n_hidden + 1,k)-1 #301x10
		max_iter = 10

		#training

		for i in range(max_iter):
			print i
			for j in range(N):
				x = [1]
				x.extend(x_train[j])
				x = np.matrix(x) #1x785
				
				h = x*w1 #1x300
				h = self.sigmf(h)
				hidden_output = np.concatenate((np.matrix([1]),np.matrix(h)), axis = 1) #1x301
				
				output = hidden_output*w2 #1x10
				output = self.sigmf(output) #1x10
				
				yd = [0 for i in range(10)]
				yd[y_train[j]] = 1
				yd = np.matrix(yd)
				#backpropagation
				d32 = np.multiply(output,1-output)
				delta32 = np.multiply(output - yd,d32)#1x10
				d21 = np.multiply(hidden_output,1-hidden_output)#1x301
				d21w = delta32*(w2.transpose()) #1x301
				delta21 = np.multiply(d21w,d21)#1x301
				delta21 = delta21[0::1,1::1]#1x300

				w1 = w1 - alpha*(x.transpose()*delta21)   #785x300
				w2 = w2 - alpha*(hidden_output.transpose()*delta32)  #301x10


		return w1,w2



		end = time.time()

	def test(self,w1,w2):

		mndata = MNIST('mnist_data')
		x_test,y_test = mndata.load_testing()	#X = 10000x784; Y = 10000x1	

		w1 = self.discretize(w1)
		w2 = self.discretize(w2)
		beta = 0.01
		N = len(x_test) #number of testing samples
		D = len(x_test[0])
		n_hidden = 300
		k = 10
		max_iter = 10
		num_correct = 0
		#testing

		
		for j in range(N):
			x = [1]
			x.extend(x_test[j])
			x = np.matrix(x) #1x785
			
			h = x*w1 #1x300
			h = self.sigmf(h)
			hidden_output = np.concatenate((np.matrix([1]),np.matrix(h)), axis = 1) #1x301
			
			output = hidden_output*w2 #1x10
			output = self.sigmf(output) #1x10

			if(np.argmax(output) == y_test[j]):
				num_correct +=1
		return num_correct,N

	def testCompNet(self,w1,w2):
		'''
			w1 is 785x300
			w2 is 301x10
		'''
		mndata = MNIST('mnist_data')
		x_test,y_test = mndata.load_testing()	#X = 10000x784; Y = 10000x1	

		w1 = self.discretize(w1)
		w2 = self.discretize(w2)
		t1 = time.time()
		print 'x1',t1
		permutation_w1 = self.get_permutation(w1)	#since hidden layers become permutation invariant
		permutation_w2 = self.get_permutation(w2)
		t2 = time.time() - t1
		print 'x2',t2
		pw1 = self.probability(w1,3)
		pw2 = self.probability(w2,3)
		t3 = time.time()-t2
		print 'x3',t3
		M_w1 = w1.shape[1]
		M_w2 = w2.shape[1]
		w1 = w1.transpose() #300x785
		w2 = w2.transpose() #10x301
		w1 = w1.tolist()
		w2 = w2.tolist()
		t4 = time.time()-t3
		print 'x4',t4

		node_w1 = Node(M_w1,3,-1)
		Compressor().formTree(node_w1,w1,0,3)
		print 'Formed Tree :)'
		l_w1,h_w1 = Compressor().compressTree(node_w1,0,1,pw1,M_w1)
		t5 = time.time()-t4
		print 'Compressed Tree :)'
		print 'x5',t5
		node_w2 = Node(M_w2,3,-1)
		Compressor().formTree(node_w2,w2,0,3)
		print 'Formed Tree :)'
		l_w2,h_w2 = Compressor().compressTree(node_w2,0,1,pw2,M_w2)
		t6 = time.time()-t5
		print 'Compressed Tree :)'
		print 'x6',t6
		beta = 0.01
		N = len(x_test) #number of testing samples
		D = len(x_test[0])
		n_hidden = 300
		k = 10
		max_iter = 10
		num_correct = 0
		#testing
		t7 = time.time()-t6
		print 'x7',t7
		for j in range(N):
			print j
			x = [1]
			x.extend(x_test[j])
			x = np.matrix(x) #1x785
			
			h = DNNInference().inferenceNN(x,l_w1,h_w1,n_hidden,pw1,D+1,0 ) #1x300
			h = self.sigmf(h)
			h = self.get_correct_permutation(h,permutation_w1)
			hidden_output = np.concatenate((np.matrix([1]),np.matrix(h)), axis = 1) #1x301
			
			output = DNNInference().inferenceNN(h,l_w2,h_w2,k,pw2,n_hidden+1,0 ) #1x10
			output = self.sigmf(output) #1x10
			output = self.get_correct_permutation(output,permutation_w2)
			if(np.argmax(output) == y_test[j]):
				num_correct +=1

		t8 = time.time()-t7
		print 'x8',t8
		return num_correct,N		


		
	def get_correct_permutation(self,h,p):

		h = [h]
		h.append(p)
		h = np.matrix(h)
		h = h.transpose()
		h = h.tolist()
		h.sort(key = lambda x : x[1])
		h = np.matrix(h)
		h = h[0::1,0:1:1]
		h = h.transpose()
		return h




	def probability(self,w,k):

		l = k/2
		pw = [0 for i in  xrange(-l,l+1)]
		
		d = w.shape[0]*w.shape[1]
		for i in xrange(-l,l+1):

			counti = np.count_nonzero(w == i)
			pw[i+l] = f(counti,d)
			d = d - counti

		return pw

	def get_permutation(self,w):
		
		length = w.shape[1]
		permutation_w = [i for i in range(length)] #since hidden layers become permutation invariant
		w = w.tolist()
		w.append(permutation_w)
		w = np.matrix(w)
		w = w.transpose()
		w = w.tolist()
		w.sort(key = lambda x : x[0:length])
		w = np.matrix(w)
		permutation_w = w[0::1,length::1]
		permutation_w = permutation_w.transpose()
		permutation_w = permutation_w.tolist()

		return permutation_w

	def discretize(self,w):

		w = np.int64(np.floor(np.maximum(-1, np.minimum(1,w))))
		return w


	def sigmf(self, h):

		ans = np.exp(-0.01*h)
		return 1/(1+ans)
''' 
To load .mat file in python
	def matToPy(self, data_name): #works only for the specific data used in this code, use scipy.io.loadmat in general
		data = open(data_name,'r')
		for i in range(5):
			data.next()
		temp = data.next()
		tempw = temp[1:len(temp)-1]
		tempsplit = tempw.split(' ')
		w = np.matrix([int(i) for i in tempsplit])
		temp = data.next()
		while temp!='\n': #stops when 'StopIteration' occurs in data
			tempw = temp[1:len(temp)-1]
			tempsplit = tempw.split(' ')
			tempW = np.matrix([int(i) for i in tempsplit])
			w = np.append(w,tempW,axis = 0)
			temp = data.next()

		return w
'''
	

