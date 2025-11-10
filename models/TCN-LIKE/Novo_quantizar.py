import os, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models

# ========= CONFIG =========
ARQ_DADOS      = "dados_treino.npz"            # precisa ter X (N,L) ou (N,L,1) e y (N,) ou (N,1)
PASTA_SAIDA    = "modelo_mcu_tcn_int8_ds64_ok"
ARQ_KERAS      = os.path.join(PASTA_SAIDA, "modelo_mcu_tcn.keras")
ARQ_TFLITE     = os.path.join(PASTA_SAIDA, "modelo_mcu_tcn_int8.tflite")
ARQ_HEADER     = os.path.join(PASTA_SAIDA, "model_data.h")
DECIM          = 64       # fator de downsample fora do grafo (64 ou 128)
F1, F2         = 8, 8     # nº de filtros leves (mantém RAM baixa)
K              = 5        # kernel temporal (5x1)

os.makedirs(PASTA_SAIDA, exist_ok=True)

# ========= DADOS =========
ds = np.load(ARQ_DADOS)
X = ds["X"]; y = ds["y"]
if X.ndim == 2: X = X[..., None]    # (N,L) -> (N,L,1)
N, L, C = X.shape
assert C == 1, f"Esperado canal único, veio {C}"

# Normalização (use a mesma em produção)
X_mean = np.mean(X, axis=(0,1), keepdims=True)
X_std  = np.std(X, axis=(0,1), keepdims=True) + 1e-8
Xn = (X - X_mean) / X_std

# ========= DOWNSAMPLE FORA DO GRAFO =========
# Assim evitamos um AveragePool enorme como 1ª camada (que podia puxar ops dinâmicas).
Ld = int(np.ceil(L / DECIM))
Xdn = Xn[:, ::DECIM, :]                 # (N, Ld, 1)

# ========= ARQUITETURA ESTÁTICA (sem ops dinâmicas) =========
def build_tcn_mcu_static(input_len: int):
    inp = layers.Input(shape=(input_len, 1), name="inp_dn")        # (Ld,1)
    x   = layers.Reshape((input_len, 1, 1), name="to_2d")(inp)     # (H,W,C)=(Ld,1,1)

    # SeparableConv2D -> Depthwise Kx1 + Pointwise 1x1
    x = layers.SeparableConv2D(F1, (K,1), padding="same", activation="relu", name="b1_sep")(x)
    x = layers.SeparableConv2D(F2, (K,1), padding="same", activation="relu", name="b2_sep")(x)

    # Reduz H -> 1 com pool FIXO (constante): evita MEAN/SHAPE/PACK
    x = layers.AveragePooling2D(pool_size=(input_len,1), strides=(input_len,1), padding="valid", name="pool_to1")(x)

    # (1,1,F2) -> vetor FIXO (sem Flatten dinâmico)
    x = layers.Reshape((F2,), name="flatten_static")(x)

    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    return models.Model(inp, out, name="tcn_mcu_int8_ds")

model = build_tcn_mcu_static(Ld)
model.compile(optimizer="adam", loss="mse")
model.save(ARQ_KERAS)
print(model.summary())

# ========= QUANTIZAÇÃO INT8 FULL-INTEGER =========
def representative_dataset_gen(n=200):
    idx = np.random.choice(N, size=min(n, N), replace=False)
    for i in idx:
        yield [Xdn[i:i+1].astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative_dataset_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
try:
    conv._experimental_new_quantizer = True
except Exception:
    pass

tflm = conv.convert()
open(ARQ_TFLITE, "wb").write(tflm)
print("✅ Gerado INT8:", ARQ_TFLITE, "| bytes =", len(tflm))

# ========= HEADER PARA ARDUINO =========
def write_header_tflite_to_c(path_tflite, path_header="model_data.h", sym="g_model"):
    b = open(path_tflite, "rb").read()
    with open(path_header, "w") as f:
        f.write('#pragma once\n#include <stdint.h>\n#include <stddef.h>\n')
        f.write('alignas(16) const unsigned char %s[] = {' % sym)
        f.write(','.join(str(x) for x in b))
        f.write('};\nconst int %s_len = %d;\n' % (sym, len(b)))
write_header_tflite_to_c(ARQ_TFLITE, ARQ_HEADER, "g_model")
print("✅ Header gerado:", ARQ_HEADER)

# ====== DICA IMPORTANTE PARA PRODUÇÃO ======
# No firmware, se você for alimentar dados reais, faça o mesmo pré-processamento:
# 1) normalizar (x - X_mean)/X_std  (grave esses valores ou normalize previamente)
# 2) dizimar por DECIM: x_dn = x[::DECIM]
