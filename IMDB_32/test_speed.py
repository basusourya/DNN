from compressor_32 import compress as c
from test_inference_32 import inference as i

#comp = c()
inf = i()

#comp.compress_network("/content/DNN/quantized_model_64.h5")
inf.compressed_inference("quantized_model_imdb_32.h5")