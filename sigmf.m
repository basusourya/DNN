function y = sigmf(x,b)
u = 1 + exp(-b(1)*(x-b(2)));
y = 1./u;