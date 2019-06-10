function [l, h, p_vec] = binomial_encode(l, h, n, p_w, N)
  %p_w is the probability of the given colour of weight
  ph = (1-p_w)^N;
  pl = 0;
  r = 0;
  p_vec = [];
  p_index = (1-p_w)^N;
  p_vec = [p_vec;p_index];
  range = h - l;
  while r != N
   r = r + 1;
   
   p_index = ((N-r+1)/r)*(p_w/(1-p_w))*p_index;
   pl = ph;
   ph = ph + p_index;
   p_vec = [p_vec;p_index]; 
  end
  
  h = l + range*ph;
  l = l + range*pl;
  
  
  
  
  
  
  
  