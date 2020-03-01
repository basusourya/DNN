from compressor_64 import compress as c
from test_inference_64 import inference as i

#comp = c()
inf = i()

#comp.compress_network("quantized_model_64.h5")
inf.compressed_inference("quantized_model_64.h5")