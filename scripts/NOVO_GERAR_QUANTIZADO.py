import os
import numpy as np
import tensorflow as tf
import pickle

# Caminhos
pasta_modelo = "modelo_final_quantizado"
modelo_keras_path = os.path.join(pasta_modelo, "modelo_final.keras")
pca_path = os.path.join(pasta_modelo, "pca.pkl")
X_pca_path = os.path.join(pasta_modelo, "X_pca_final.npy")
tflite_saida = os.path.join(pasta_modelo, "modelo_final_quantizado.tflite")

print("📦 Recarregando modelo e dados salvos...")

# Carregar modelo Keras já treinado
model = tf.keras.models.load_model(modelo_keras_path)

# Carregar dados para calibragem
X_pca = np.load(X_pca_path)

# Função de calibragem para TFLite
def representative_dataset():
    for i in range(min(200, len(X_pca))):
        yield [X_pca[i:i+1].astype(np.float32)]

# Conversor TFLite com quantização e calibragem
print("⚙️ Reconvertendo para TFLite com quantização real...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # Opcional: INT8 puro
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# Gerar e salvar modelo TFLite
tflite_model = converter.convert()
with open(tflite_saida, "wb") as f:
    f.write(tflite_model)

print(f"✅ Novo modelo quantizado TFLite salvo com sucesso em:\n→ {tflite_saida}")
