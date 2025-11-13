# build_readout_int8.py
import numpy as np, tensorflow as tf

# ---- Carrega pesos do Ridge ----
W = np.load("coeficientes_ridge.npy")   # (500,) ou (500,1)
b = np.load("bias_ridge.npy")           # escalar ou (1,)

W = W.reshape(-1, 1).astype(np.float32)         # (500,1)
b = np.array([float(np.squeeze(b))], np.float32)  # (1,)

D = W.shape[0]  # dimensão de entrada (ex.: 500)

# ---- Keras: Dense(1) => mapeará para FullyConnected no TFLite ----
inp = tf.keras.Input(shape=(D,), name="reservoir_states")
out = tf.keras.layers.Dense(1, use_bias=True, name="readout")(inp)
model = tf.keras.Model(inp, out)
model.get_layer("readout").set_weights([W, b])

# ---- Representativo p/ quantização ----
repr_states = np.load("repr_states.npy")  # shape (N, D)

def rep_ds(n=200):
    n = min(n, len(repr_states))
    for i in range(n):
        yield [repr_states[i:i+1].astype(np.float32)]

# ---- Converter p/ INT8 (apenas builtins) ----
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_ds
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8

tflm = conv.convert()
open("readout_int8.tflite", "wb").write(tflm)
print("OK - TFLite gerado:", len(tflm), "bytes")
