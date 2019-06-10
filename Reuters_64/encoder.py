from fractions import Fraction as f
import binstr
from math import ceil, floor

class encoders(object):
	
	def binomial_encoder_frequencies(self, freq, N):
		
		#print('symbol', symbol)
		#print('N:',N)
		a = [0]*(N+1) #last one for EOF
		total_freq = sum(freq)
		pw = f(freq[0], total_freq)
		if pw == 1:
			a[N] = 100000	
			return a

		ph = (1-pw)**N
		a[0] = int(ph*100000)
		pl = f(0,1)
		r = 0
		pindex = ph
		for i in range(N):
			r = r + 1
			pindex = f((N-r+1),r)*f(pw,(1-pw))*pindex 
			a[i+1] = int(pindex*100000)
		#if a[symbol] == 0:
		#	a[symbol] = 1
		a = [x+1 for x in a]
		return a
		#text_file = open("Output.txt", "a")
		#text_file.write(binstr.frac_to_b(float((result))))
		#text_file.close()


	def std_encoder(self,p, k): #p: cummulative probability should start with 0 and end at 1, with k+1 elements
		binary_code = self.get_binary_code(f(p[k]),f(p[k+1]))
		return binary_code
		
	def nextHigherInteger(self, l):
		a = ceil(l)
		if l == a:
			a += 1

		return a

	def prevLowerInteger(self, h):
		a = floor(h)
		if h == a:
			a -= 1

		return a

	def get_binary_code(self, l, h): 
	
		d = 1
		len_d = 0
		x = self.prevLowerInteger(h)

		while x < l:
			
			l *= 2
			h*= 2
			d *= 2
			len_d += 1
			x = self.prevLowerInteger(h)
		if x == 0:
			len_d+=1
		return self.dyadic_to_binary(x,len_d)

	def dyadic_to_binary(self, x, len_d):

		s = ''
		len_s = 0
		
		while x>0:
			rem = x%2
			s = str(rem) + s
			x = x//2
			len_s += 1

		while len_d > len_s:
			s = '0' + s
			len_d -= 1

		return s



	def cut_string(self, s):

		flag = 0
		size = len(s)
		for i in range(len(s) - 1):

			if s[size - i -1] == '1':

				flag = 1

			if flag == 1:
				break


		if flag == 1:
			s = s[0:size - i]
		else:
			s = s[0:1]

		return s
