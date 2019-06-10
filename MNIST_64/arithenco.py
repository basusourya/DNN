# 
# Compression application using static arithmetic coding
# 
# Usage: python arithmetic-compress.py InputFile OutputFile
# Then use the corresponding arithmetic-decompress.py application to recreate the original input file.
# Note that the application uses an alphabet of 257 symbols - 256 symbols for the byte
# values and 1 symbol for the EOF marker. The compressed file format starts with a list
# of 256 symbol frequencies, and then followed by the arithmetic-coded data.
# 
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# 

import sys
import arithmeticcoding
python3 = sys.version_info.major >= 3

class arithencoder(object):
# Command line main application function.
	s = []
	f = []
	def main(self, inp, frequencies, num_symbols):
		# Handle command line arguments
		#if len(args) != 2:
		#	sys.exit("Usage: python arithmetic-compress.py InputFile OutputFile")
		#inputfile  = args[0]
		#outputfile = args[1]
	
		# Read input file once to compute symbol frequencies
		freqs = self.get_frequencies(inp, frequencies, num_symbols)
		#freqs.increment(num_symbols)  # EOF symbol gets a frequency of 1
	
		# Read input file again, compress with arithmetic coding, and write output file
		#with open(inputfile, "rb") as inp:
		#	bitout = arithmeticcoding.BitOutputStream(open(outputfile, "wb"))
		#	try:
		#		write_frequencies(bitout, freqs)
		self.s = self.compress(inp, freqs, num_symbols)
		#	finally:
		#		bitout.close()
		return self.s
	

	# Returns a frequency table based on the bytes in the given file.
	# Also contains an extra entry for symbol 256, whose frequency is set to 0.
	def get_frequencies(self, inp, frequencies, num_symbols):
		freqs = arithmeticcoding.SimpleFrequencyTable(frequencies)
		#self.f = [0 for i in range(num_symbols+1)]
		#for i in range(len(inp)):
		#	b = inp[i]
		#	freqs.increment(b)
		#	self.f[b] += 1
		#self.f[num_symbols] += 1
		#self.f = frequencies
		return freqs


#	def write_frequencies(bitout, freqs):
#		for i in range(256):
#			write_int(bitout, 32, freqs.get(i))


	def compress(self, inp, freqs, num_symbols):
		enc = arithmeticcoding.ArithmeticEncoder()
		for i in range(len(inp)):
			symbol = inp[i]
			enc.write(freqs, symbol)
		enc.write(freqs, num_symbols)  # EOF
		self.s = enc.finish()  # Flush remaining code bits
		return self.s


	# Writes an unsigned integer of the given bit width to the given stream.
#	def write_int(bitout, numbits, value):
#		for i in reversed(range(numbits)):
#			bitout.write((value >> i) & 1)  # Big endian


	# Main launcher
	#if __name__ == "__main__":
	#	main(sys.argv[1 : ])