function W = discretize(W)
W = floor(max(-63,min(64,W)));