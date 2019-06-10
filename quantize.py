
#accuracy of 93.83 with 101 quantization levels
from utils import utilClass as uc
class main(object):

	def fQ(self, weights, steps):

		
		for i in range(len(weights)):
			for j in range(len(weights[0])):
				weights[i][j] = self.quanitze_steps(weights[i][j], steps)

		return weights


	def quanitze_steps(self, k, steps):

		bs = uc()
		return bs.b_search(steps, k)

