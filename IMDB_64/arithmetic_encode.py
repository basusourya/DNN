class encoder(object):

	Code_value_bits = 32
	Top_value = (1<<Code_value_bits)-1
	First_qtr = Top_value/4+1
	Half	  = 2*First_qtr	
	Third_qtr = 3*First_qtr
	low = 0
	high = Top_value	
	bits_to_follow = 0
	s = ''

	def start_encoding(self):

		self.low = 0
		self.high = self.Top_value
		self.bits_to_follow = 0

	def encode_symbol(self, symbol, cum_freq):

		range_1 = int(self.high - self.low) + 1
		length = len(cum_freq)
		self.high = self.low +int((range_1*cum_freq[symbol+1])/cum_freq[length-1])-1
		self.low = self.low + int((range_1*cum_freq[symbol])/cum_freq[length-1])
		print (self.low, self.high, symbol)
		while 1:
			
			if self.high < self.Half:
				self.bits_plus_follow(0)

			elif self.low >= self.Half:
				self.bits_plus_follow(1)
				self.low -= self.Half
				self.high -= self.Half

			elif self.low >= self.First_qtr and self.high <= self.Third_qtr:
				self.bits_to_follow += 1
				self.low -= self.First_qtr
				self.high -= self.First_qtr

			else:
				break

			self.low = int(2*self.low)
			self.high = int(2*self.high) + 1


	def done_encoding(self):

		self.bits_to_follow += 1

		if self.low < self.First_qtr:
			self.bits_plus_follow(0)

		else:
			self.bits_plus_follow(1)

		return self.s


	def bits_plus_follow(self,bit):

		self.s = self.s + str(bit)

		while  self.bits_to_follow > 0:
			self.s = self.s + str(1-bit)
			self.bits_to_follow -= 1 
