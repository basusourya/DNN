from fractions import Fraction as f

class decoders(object):

	def binomial_decoder(self, l, h , pw, N):
		if pw == 1:
			return l,h,N
		ph = (1-pw)**(N)
		pl = f(0,1)
		r = 0
		ans = 0
		flag = 0
		if l>=pl and h<=ph:
			ans = r
			flag = 1
		pindex = ph
		range = h - l
		while r != N and flag is 0:
			r = r + 1
			pindex = f(N-r+1,r)*f(pw,1-pw)*pindex
			pl = ph
			ph = ph + pindex
			if l>=pl and h<=ph:
				ans = r
				flag = 1
		h = (h-pl)/(ph-pl)
		l = (l-pl)/(ph-pl)
		return l,h,ans
