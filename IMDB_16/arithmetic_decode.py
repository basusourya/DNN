class decoder(object):

	Code_value_bits = 32
	Top_value = (1<<Code_value_bits)-1
	First_qtr = Top_value/4+1
	Half	  = 2*First_qtr	
	Third_qtr = 3*First_qtr
	value = 0
	low = 0
	high = Top_value
	s = ''

	def start_decoding(self,s):

		self.value = 0
		self.s = s
		for i in range(self.Code_value_bits):

			self.value = int(2*self.value) + self.input_bit()

		self.low = 0
		self.high = self.Top_value


	def decode_symbol(self, cum_freq):

		length = len(cum_freq)
		range_1 = int(self.high - self.low) + 1
		cum = int(((int(self.value - self.low) + 1)*cum_freq[length-1] - 1)/range_1)

		symbol = 0
		for i in range(length-1):

			if cum > cum_freq[symbol] and cum <= cum_freq[symbol+1]:
				break
			symbol+=1
		self.high = self.low +int((range_1*cum_freq[symbol+1])/cum_freq[length-1])-1
		self.low = self.low + int((range_1*cum_freq[symbol])/cum_freq[length-1])
			
		print(cum, symbol, self.low, self.high)
		while 1:
			print (self.low, self.high)
			if self.high < self.Half:
				'''Nothing to do'''
				#print('x')
			elif self.low >= self.Half:
				self.value -= self.Half
				self.low -= self.Half
				self.high -= self.Half
				#print('y')

			elif self.low >= self.First_qtr and self.high <self.Third_qtr:
				self.value -= self.First_qtr
				self.low -= self.First_qtr
				self.high -= self.First_qtr
				#print('z')

			else:
				#print('w')
				break

			self.low = int(2*self.low)
			self.high = int(2*self.high) + 1
			self.value = int(2*self.value)+self.input_bit()

		return symbol, len(self.s)

	def input_bit(self):

		a = self.s[0]
		self.s = self.s[1:]
		return int(a)

