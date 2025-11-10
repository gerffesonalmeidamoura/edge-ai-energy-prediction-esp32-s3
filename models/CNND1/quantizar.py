#!/usr/bin/env python3
# convert_and_evaluate_tflite.py

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# === CONFIGURAÇÕES ===
PASTA_MODELO    = "resultado_treinamento_142560"
PASTA_SAIDA     = "modelo_final_quantizado"
KERAS_MODEL_FP  = os.path.join(PASTA_MODELO, "modelo_final.keras")
TFLITE_MODEL_FP = os.path.join(PASTA_SAIDA, "modelo_final.tflite")
ARQUIVO_DADOS   = "dados_treino.npz"
METRICS_FP      = os.path.join(PASTA_SAIDA, "metricas_quantizado.txt")

os.makedirs(PASTA_SAIDA, exist_ok=True)

# === CONFIGURAR GPU (memória controlada) ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU configurada com uso de memória controlado.")
    except RuntimeError as e:
        print(f"⚠️ Erro ao configurar GPU: {e}")
else:
    print("⚠️ Nenhuma GPU disponível. Usando CPU.")

# === 1) Carregar modelo Keras original ===
print(f"\n📦 Carregando modelo Keras: {KERAS_MODEL_FP}")
model = tf.keras.models.load_model(KERAS_MODEL_FP)

# === 2) Converter para TFLite ===
print("🔧 Convertendo para TFLite (float32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Se desejar quantização, descomente as linhas abaixo:
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# def representative_dataset():
#     # gerador de exemplo: for x in X_train[:100]: yield [x.astype(np.float32)]
#     ...
# converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# === 3) Salvar modelo TFLite ===
with open(TFLITE_MODEL_FP, "wb") as f:
    f.write(tflite_model)
print(f"✅ Modelo TFLite salvo em: {TFLITE_MODEL_FP}")

# === 4) Carregar dados para avaliação ===
print(f"\n📥 Carregando dados de avaliação: {ARQUIVO_DADOS}")
data = np.load(ARQUIVO_DADOS)
X, y = data["X"], data["y"]
print(f"✅ Dados carregados → X: {X.shape}, y: {y.shape}")

# === 5) Inferência com TFLite ===
print("\n🧪 Iniciando inferência com TFLite...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_FP)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

y_pred = []
for i in tqdm(range(len(X)), desc="Inferindo"):
    inp = X[i:i+1].astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])[0][0]
    y_pred.append(float(out))

y_pred = np.array(y_pred, dtype=np.float32)

# === 6) Cálculo de métricas ===
mae  = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2   = r2_score(y, y_pred)
err_pct = np.mean(np.abs((y - y_pred) / y)) * 100

# === 7) Salvar métricas em arquivo ===
with open(METRICS_FP, "w") as f:
    f.write(f"MAE:  {mae:.2f} kWh\n")
    f.write(f"RMSE: {rmse:.2f} kWh\n")
    f.write(f"R²:   {r2:.4f}\n")
    f.write(f"Erro Percentual Médio: {err_pct:.2f}%\n")

print(f"\n✅ Métricas salvas em: {METRICS_FP}")
print(f"   MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f} | Erro%: {err_pct:.2f}%")
