This folder contains the code for the paper "Universal and Succinct Source Coding of Deep Neural Networks", https://arxiv.org/pdf/1804.02800.pdf

Important files: 

- quantized_mnist.py: Gives quantized model as output.
- NNCompressor.py: Contains the compressor algorithm.
- inference.py: Contains the inference algorithm.
- test_inference.py: Combines different files to implement the compressed inference task in the 'compressed_inference' function of the class 'inference'.
- compressor.py:  Takes functons from NNCompressor.py and implements a high level algorithm to compress the quantized_model.

This set of codes demonstrates the compressed inference of our algorithm. The codes involves functionalities including training of a DNN, compressing it, 
and then making a compressed inference using the compressed network without completely decompressing it.
Flow of the code:

1) Train the model and get the quantized model saved as 'quantized_model.h5' by running quantized_mnist.py for the mlp with the mnist test data.
2) To make compressed inference using the saved mnist model, use the compressed_inference function in the inference class of the file test_inference which takes
   as input the name of the saved model, i.e. 'quantized_mnist'.
   
