from compressor_16 import compress as c
from test_inference_16 import inference as i

#comp = c()
inf = i()

#comp.compress_network("/content/DNN/quantized_model_64.h5")
inf.compressed_inference("quantized_model_16.h5")