from encoder import encoders as e
from arithmetic_encode import encoder
from arithmetic_decode import decoder
from arithenco import arithencoder as ec
from arithdeco import arithdecoder as dc
import numpy as np 
import binstr
from math import log2
from fractions import Fraction as f
import time
from math import log2
class test_speed(object):
	
	def test_binomial_encoder(self, size):

		N = 2
		pw = f(0.3)
		cumm_p = [f(0),f(0.3),f(1)]
		a = np.random.uniform(low = 0, high = 1, size = size)
		count = [0,0]
		for i in range(len(a)):
			if a[i] < 0.3:
				a[i] = 0
				count[0] += 1
			else:
				a[i] = 1
				count[1] += 1

		cum_freq = [0,0,0]
		temp = 0
		for i in range(2):

			cum_freq[i+1] = count[i] + temp
			temp = cum_freq[i+1]

		print(count)
		print(cum_freq)
		b = [int(a[i]) for i in range(int(len(a)))]
		s, frequencies = ec().main(b, 2)

		length = len(s)
		c = dc().main(s, frequencies, 2)
		flag = 1
		print(len(c))
		for i in range(len(b)):
			if c[i] != b[i]:
				print (i,'Unfortunately, the code was not decoded properly!')
				flag = 0
		if flag:
			print ('Nice Work!')
	

		print ('length:', length)
		print ('frequencies', frequencies)


	def test_std_encoder(self, size):

		N = 3
		pw = f(0.3)
		cumm_p = [f(0),f(0.1),f(0.7),f(0.95),f(1)]
		a = np.random.uniform(low = 0, high = 1, size = size)
		count = [0,0,0,0]
		for i in range(len(a)):
			if a[i] < 0.1:
				a[i] = 0
				count[0] += 1
			elif a[i] >= 0.1 and a[i] < 0.7:
				a[i] = 1
				count[1] += 1
			elif a[i] >= 0.7 and a[i] < 0.95:
				a[i] = 2
				count[2] += 1
			else:
				a[i] = 3
				count[3] += 1
		cum_freq = [0,0,0,0,0]
		temp = 0
		freqs = [10000,60000,25000,5000,1]
		for i in range(4):

			cum_freq[i+1] = count[i] + temp
			temp = cum_freq[i+1]
		count.append(1)
		print(count)
		print(cum_freq)
		b = [int(a[i]) for i in range(int(len(a)))]
		#text_file = open("Output_std.txt", "a")
		length = 0
		start_time = time.time()
		#ec = encoder()
		#ec.start_encoding()
		#for i in range(len(b)):
			
			#ec.encode_symbol(b[i], cum_freq)

		s = ec().main(b, freqs, 4)
		length = len(s)
		time_elapsed = time.time() - start_time
		c = dc().main(s, freqs, 4)
		flag = 1
		print(len(c))
		for i in range(len(b)):
			if c[i] != b[i]:
				print (i,'Unfortunately, the code was not decoded properly!')
				flag = 0
		if flag:
			print ('Nice Work!')
		print('Time elapsed: ', time_elapsed)

		print ('length:', length)
		#print ('frequencies', frequencies)





	def test_arithenco(self, size):
		
		pw = f(0.3)
		cumm_p = [f(0),f(0.1),f(0.7),f(0.95),f(1)]
		a = np.random.uniform(low = 0, high = 1, size = size)
		count = [0,0,0,0]
		text_file = open("Input.txt", "w")
		for i in range(len(a)):
			if a[i] < 0.1:
				a[i] = 0
				x = 0
				count[0] += 1
			elif a[i] >= 0.1 and a[i] < 0.7:
				a[i] = 1
				x = 1
				count[1] += 1
			elif a[i] >= 0.7 and a[i] < 0.95:
				a[i] = 2
				count[2] += 1
				x = 2
			else:
				a[i] = 3
				count[3] += 1
				x = 3
			text_file.write(str(x+1))
			#text_file.write('\n')
		text_file.close()


	def test_decoded(self):

		t_1 = open("Input.txt", "r")
		t_2 = open("Input_decoded.txt", "r")
		#x = t_1.read(1)
		i = 0
		while 1:
			x = t_1.read(1)
			print(x)
			if x == '':
				break
			if x!=t_2.read(1):
				print('Not Correctly decoded')
				break
			i+=1
		print('Correctly Decoded')
		
	def calculate_entropy(self, p):
		entropy = 0
		for i in range(len(p)):

			entropy = entropy - p[i]*log2(p[i])

		return entropy