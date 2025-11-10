# quantizar_seguro_v3_parallel.py
import os, json, hashlib, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from reservoirpy.nodes import Reservoir
import joblib
from joblib import Parallel, delayed

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===========================
# CONFIG
# ===========================
MODEL_DIR   = r"C:\projeto_artigo\mono\2_ESN\com_interharmonicas\modelo_treinado_seguro_v3"
CSV_DIR     = r"C:\projeto_artigo\mono\TCN\com_interharmonicas\amostras_3dias_sem_harmonicas"
OUT_DIR     = r"C:\projeto_artigo\mono\2_ESN\com_interharmonicas\modelo_treinado_seguro_v3_quantizado"

TARGET_LEN  = 142560
UNITS       = 1000
SR          = 1.0
NORMALIZE   = True

N_JOBS      = 8              # multiprocessamento
REPR_FRAC   = 0.20             # ~20% dos arquivos para calibração
REPR_MIN    = 80               # mínimo
REPR_MAX    = 300              # máximo (pra não ficar lento)
RNG_SEED    = 123              # reprodutibilidade

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================
# Utils: fingerprint, colunas, ESN, quant/dequant
# ===========================
def fingerprint_res(units, sr, d_inputs):
    rng = np.random.default_rng(123)
    U = rng.standard_normal((512, d_inputs)).astype(np.float32)
    U = (U - U.mean())/(U.std()+1e-8)
    res = Reservoir(units=units, sr=sr, input_scaling=1.0, bias_scaling=0.0, seed=42)
    res.reset()
    s = res.run(U)[-1].astype(np.float32)
    return hashlib.sha256(s.tobytes()).hexdigest()

def padronizar_cols(df, ref_cols):
    drop_tokens = ['energy','accum','kwh']
    cols_drop = [c for c in df.columns if any(t in c.lower() for t in drop_tokens)]
    if cols_drop: df = df.drop(columns=cols_drop)
    df = df[[c for c in ref_cols if c in df.columns]]
    if len(df.columns) != len(ref_cols):
        raise ValueError("Colunas do CSV não batem com as do treino.")
    return df

def last_state_from_csv(csv_path, ref_cols):
    df = pd.read_csv(csv_path)
    df = padronizar_cols(df, ref_cols)
    x  = df.values.astype(np.float32)
    x  = resample(x, TARGET_LEN).astype(np.float32)
    if NORMALIZE:
        x = (x - x.mean())/(x.std()+1e-8)
    res = Reservoir(units=UNITS, sr=SR, input_scaling=1.0, bias_scaling=0.0, seed=42)
    res.reset()
    state = res.run(x)[-1].astype(np.float32)   # (UNITS,)
    return state

def compute_states_parallel(files, ref_cols, desc):
    results = Parallel(n_jobs=N_JOBS, backend="loky", prefer="processes")(
        delayed(last_state_from_csv)(os.path.join(CSV_DIR, f), ref_cols)
        for f in tqdm(files, desc=desc)
    )
    states = [r for r in results if r is not None]
    return np.stack(states, axis=0).astype(np.float32)

def representative_dataset_from_array(states):
    for i in range(states.shape[0]):
        yield [states[i:i+1]]  # (1, UNITS) float32

def write_cc_from_tflite(tflite_bytes, out_cc, var_name="g_model_data"):
    with open(out_cc, "w") as f:
        f.write("// Auto-generated TFLite Micro model data\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"const unsigned char {var_name}[] = {{\n  ")
        for i, b in enumerate(tflite_bytes):
            f.write(f"{b},")
            if (i+1) % 12 == 0:
                f.write("\n  ")
        f.write("\n};\n")
        f.write(f"const int {var_name}_len = {len(tflite_bytes)};\n")

def quantize_fp32_to_int8(x, scale, zp):
    q = np.round(x / scale + zp).astype(np.int8)
    return np.clip(q, -128, 127).astype(np.int8)

def dequantize_int8_to_fp32(q, scale, zp):
    return scale * (q.astype(np.int32) - zp)

# ===========================
# 1) Carrega artefatos do treino seguro
# ===========================
ridge_model = joblib.load(os.path.join(MODEL_DIR, "ridge_model.pkl"))
with open(os.path.join(MODEL_DIR, "columns.json"), "r", encoding="utf-8") as f:
    REF_COLS = json.load(f)
with open(os.path.join(MODEL_DIR, "reservoir_fingerprint.json"), "r") as f:
    FP_TRAIN = json.load(f)

# Checagem de fingerprint (mesmo ESN / mesmo D)
FP_TEST = {"units":UNITS, "sr":SR, "inputs":len(REF_COLS), "hash": fingerprint_res(UNITS, SR, len(REF_COLS))}
if FP_TEST != FP_TRAIN:
    raise RuntimeError(f"[ERRO] Fingerprint do reservatório difere!\nTreino: {FP_TRAIN}\nQuantização: {FP_TEST}")

# ===========================
# 2) Converte Ridge -> Keras Dense
# ===========================
input_dim = ridge_model.coef_.shape[0]
if input_dim != UNITS:
    raise RuntimeError(f"Input do modelo ({input_dim}) != UNITS configurado ({UNITS}).")

model_keras = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(1, activation="linear", use_bias=True)
])
# pesos: W shape (input_dim, 1) e bias shape (1,)
W = ridge_model.coef_.reshape(-1, 1).astype(np.float32)
b = np.array([ridge_model.intercept_], dtype=np.float32)
model_keras.layers[0].set_weights([W, b])

keras_path = os.path.join(OUT_DIR, "ridge_model_esn.keras")
model_keras.save(keras_path)
print("✔ Keras salvo em:", keras_path)

# ===========================
# 3) Representative dataset (paralelo)
# ===========================
all_csvs = sorted([f for f in os.listdir(CSV_DIR) if f.endswith(".csv")])
if not all_csvs:
    raise RuntimeError("Nenhum CSV encontrado em CSV_DIR.")

random.seed(RNG_SEED)
n_repr = min(max(int(len(all_csvs)*REPR_FRAC), REPR_MIN), REPR_MAX)
repr_files = random.sample(all_csvs, n_repr)
print(f"✔ Representative dataset: {len(repr_files)} arquivos (de {len(all_csvs)})")

repr_states = compute_states_parallel(repr_files, REF_COLS, desc="Estados (representative)")
np.save(os.path.join(OUT_DIR, "repr_states.npy"), repr_states)

# ===========================
# 4) Converter para TFLite INT8 (fully quantized)
# ===========================
converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = lambda: representative_dataset_from_array(repr_states)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
tflite_path = os.path.join(OUT_DIR, "ridge_model_esn_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print("✅ TFLite INT8 salvo em:", tflite_path)

# ===========================
# 5) Validação rápida (Keras vs TFLite INT8)
# ===========================
# usa mais alguns arquivos para teste (ou parte do repr_files se faltar)
val_pool = [f for f in all_csvs if f not in repr_files]
if not val_pool:
    val_pool = repr_files[:]

n_val = min(64, len(val_pool))
val_files = random.sample(val_pool, n_val)
val_states = compute_states_parallel(val_files, REF_COLS, desc="Estados (validação)")
np.save(os.path.join(OUT_DIR, "val_states.npy"), val_states)

# Keras FP32
y_keras = model_keras.predict(val_states, verbose=0).reshape(-1)

# TFLite INT8
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
in_scale, in_zp   = inp_det["quantization"]
out_scale, out_zp = out_det["quantization"]

Xq = quantize_fp32_to_int8(val_states, in_scale, in_zp)
y_tflite = []
for i in range(Xq.shape[0]):
    interpreter.set_tensor(inp_det["index"], Xq[i:i+1])
    interpreter.invoke()
    qout = interpreter.get_tensor(out_det["index"])
    y_tflite.append(dequantize_int8_to_fp32(qout, out_scale, out_zp)[0,0])
y_tflite = np.array(y_tflite, dtype=np.float32)

diff = np.abs(y_keras - y_tflite)
with open(os.path.join(OUT_DIR, "quant_validation.txt"), "w") as f:
    f.write(f"MAE(Keras vs INT8): {diff.mean():.6f}\nmax abs diff: {diff.max():.6f}\n")
print(f"Validação Keras vs INT8 -> MAE: {diff.mean():.6f} | max: {diff.max():.6f}")

# ===========================
# 6) Exporta .cc para TFLite Micro (ESP32)
# ===========================
cc_path = os.path.join(OUT_DIR, "model_data.cc")
write_cc_from_tflite(tflite_model, cc_path, var_name="g_model_data")
print("✅ Arquivo C gerado:", cc_path)

print("\nTudo pronto! Use `ridge_model_esn_int8.tflite` para TFLite normal")
print("ou `model_data.cc` no ESP32 (TFLite Micro).")
