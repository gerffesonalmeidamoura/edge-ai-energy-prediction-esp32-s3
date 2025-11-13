# inspect_tflite_io.py
import sys, numpy as np
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except:
    from tflite_runtime.interpreter import Interpreter

path = sys.argv[1]
itp = Interpreter(model_path=path); itp.allocate_tensors()
i = itp.get_input_details()[0]; o = itp.get_output_details()[0]
in_shape = i["shape"].tolist(); out_shape = o["shape"].tolist()
def flat_len(shape): 
    return int(np.prod(shape[1:])) if len(shape)>=2 and shape[0]==1 else int(np.prod(shape))
print("INPUT_LEN", flat_len(in_shape))
print("N_OUTPUTS", flat_len(out_shape))
