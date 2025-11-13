# GERAR_H.py
# Reconstrói o readout da ESN (Dense(1)) a partir dos coeficientes do Ridge,
# converte para TFLite sem FLEX/CUSTOM (INT8 e/ou FP16) e gera model_data.h.
#
# Uso:
#   python GERAR_H.py            # INT8 + FP16 (gera header a partir do INT8)
#   python GERAR_H.py int8       # só INT8
#   python GERAR_H.py fp16       # só FP16

import os, sys, numpy as np, tensorflow as tf
from tensorflow import keras
import tempfile, shutil
import joblib

ARQ_STATES_NPZ = "reservoir_states.npz"
ARQ_W = "coeficientes_ridge.npy"
ARQ_B = "bias_ridge.npy"
ARQ_PKL = "ridge_model.pkl"

OUT_INT8 = "readout_int8.tflite"
OUT_FP16 = "readout_fp16.tflite"
HDR_OUT  = "model_data.h"
HDR_SYM  = "g_model"

mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "both")  # int8 | fp16 | both
DO_INT8 = mode in ("int8", "both")
DO_FP16 = mode in ("fp16", "both")

def pick_states(path_npz):
    if not os.path.isfile(path_npz):
        raise FileNotFoundError(f"Não encontrei {path_npz}")
    z = np.load(path_npz)
    prefer = [
        "repr_states","val_states","states","states_train","X","X_train",
        "reservoir_states","features"
    ]
    for k in prefer:
        if k in z and z[k].ndim == 2:
            X = z[k]
            print(f"→ usando '{k}' de {path_npz}: {X.shape}")
            return np.asarray(X, np.float32)
    # fallback: 1ª matriz 2D que aparecer
    for k in z.files:
        arr = z[k]
        if arr.ndim == 2:
            print(f"→ usando '{k}' (fallback) de {path_npz}: {arr.shape}")
            return np.asarray(arr, np.float32)
    raise KeyError(f"Nenhuma matriz 2D encontrada em {path_npz}. Chaves: {list(z.files)}")

def load_wb(D):
    if os.path.isfile(ARQ_W) and os.path.isfile(ARQ_B):
        w = np.load(ARQ_W).astype(np.float32).reshape(-1)
        b = float(np.load(ARQ_B))
        if w.shape[0] != D:
            raise ValueError(f"coeficientes_ridge.npy tem {w.shape[0]} dims; esperado {D}")
        return w.reshape(D,1), np.float32(b)
    if os.path.isfile(ARQ_PKL):
        ridge = joblib.load(ARQ_PKL)
        w = np.asarray(ridge.coef_, np.float32).reshape(-1)
        if w.shape[0] != D:
            raise ValueError(f"ridge_model.pkl coef tem {w.shape[0]} dims; esperado {D}")
        return w.reshape(D,1), np.float32(ridge.intercept_)
    raise FileNotFoundError("Faltam pesos: nem .npy (coef/bias) nem ridge_model.pkl")

def build_model(D, w, b):
    x = keras.Input(shape=(D,), dtype="float32", name="reservoir_states")
    y = keras.layers.Dense(1, activation=None, name="readout")(x)
    m = keras.Model(x, y, name="esn_readout")
    m.get_layer("readout").set_weights([w.astype(np.float32), np.array([b], np.float32)])
    # assegura construção do grafo
    _ = m(np.zeros((1, D), np.float32))
    return m

def rep_ds(X, n=500):
    n = min(n, X.shape[0])
    idx = np.random.choice(X.shape[0], size=n, replace=False)
    for i in idx:
        yield [X[i:i+1].astype(np.float32)]

def convert_int8(model, X, D, out_path):
    # Caminho 1: from_keras_model
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: rep_ds(X)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type  = tf.int8
        conv.inference_output_type = tf.int8
        blob = conv.convert()
    except Exception as e1:
        # Caminho 2: from_concrete_functions
        try:
            sig = tf.TensorSpec([None, D], tf.float32, name="reservoir_states")
            concrete = tf.function(lambda t: model(t)).get_concrete_function(sig)
            conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = lambda: rep_ds(X)
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type  = tf.int8
            conv.inference_output_type = tf.int8
            blob = conv.convert()
        except Exception as e2:
            # Caminho 3: SavedModel
            try:
                tmp = tempfile.mkdtemp()
                sig = tf.TensorSpec([None, D], tf.float32, name="reservoir_states")
                @tf.function(input_signature=[sig])
                def serving(x): return model(x)
                tf.saved_model.save(model, tmp, signatures={'serving_default': serving})
                conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                conv.representative_dataset = lambda: rep_ds(X)
                conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                conv.inference_input_type  = tf.int8
                conv.inference_output_type = tf.int8
                blob = conv.convert()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    open(out_path, "wb").write(blob)
    print(f"✅ INT8 salvo: {out_path} ({len(blob)} bytes)")
    return blob

def convert_fp16(model, D, out_path):
    # Caminho 1: from_keras_model
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
        blob = conv.convert()
    except Exception as e1:
        # Caminho 2: from_concrete_functions
        try:
            sig = tf.TensorSpec([None, D], tf.float32, name="reservoir_states")
            concrete = tf.function(lambda t: model(t)).get_concrete_function(sig)
            conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
            blob = conv.convert()
        except Exception as e2:
            # Caminho 3: SavedModel
            try:
                tmp = tempfile.mkdtemp()
                sig = tf.TensorSpec([None, D], tf.float32, name="reservoir_states")
                @tf.function(input_signature=[sig])
                def serving(x): return model(x)
                tf.saved_model.save(model, tmp, signatures={'serving_default': serving})
                conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                conv.target_spec.supported_types = [tf.float16]
                blob = conv.convert()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    open(out_path, "wb").write(blob)
    print(f"✅ FP16 salvo: {out_path} ({len(blob)} bytes)")
    return blob

def write_header(tflite_path, header_path, sym):
    b = open(tflite_path, "rb").read()
    print(f"📦 Header de {len(b)} bytes -> {header_path} (símbolo {sym})")
    lines = []
    for i in range(0, len(b), 12):
        chunk = b[i:i+12]
        lines.append("  " + ", ".join(f"0x{bb:02x}" for bb in chunk) + ",")
    with open(header_path, "w", newline="\n") as f:
        f.write("#pragma once\n#include <stdint.h>\n#include <stddef.h>\n\n")
        f.write(f"// Gerado de {os.path.basename(tflite_path)}\n")
        f.write(f"alignas(16) const unsigned char {sym}[] = {{\n")
        f.write("\n".join(lines))
        f.write("\n};\n")
        f.write(f"const int {sym}_len = {len(b)};\n")
    print("✅ Header gerado:", header_path)

if __name__ == "__main__":
    X = pick_states(ARQ_STATES_NPZ)   # (N,D)
    N, D = X.shape
    w, b = load_wb(D)
    model = build_model(D, w, b)

    last = None
    if DO_INT8:
        convert_int8(model, X, D, OUT_INT8); last = OUT_INT8
    if DO_FP16:
        convert_fp16(model, D, OUT_FP16);     last = last or OUT_FP16

    if last:
        write_header(last, HDR_OUT, HDR_SYM)
