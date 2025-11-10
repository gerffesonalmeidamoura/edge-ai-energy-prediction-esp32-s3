import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Caminhos
pasta_base = r"C:\projeto_artigo\mono\2_TCN\com_interharmonicas"
arquivo_modelo = os.path.join(pasta_base, "resultado_treinamento_TCN_filtros64", "modelo_tcn.keras")
arquivo_X_treino = os.path.join(pasta_base, "X_train.npy")
arquivo_scaler = os.path.join(pasta_base, "scaler_treino.save")
saida = os.path.join(pasta_base, "modelo_final_quantizado")
os.makedirs(saida, exist_ok=True)

# Carregar X_train para calibração
print("📥 Carregando X_train para calibração...")
X_train = np.load(arquivo_X_treino).reshape(-1, 1)

# Ajustar scaler (idêntico ao usado no treinamento)
print("⚙️ Aplicando StandardScaler...")
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, arquivo_scaler)  # Salvar scaler para uso futuro

# Recarregar X_train normalizado para shape original
X_train = scaler.transform(X_train).reshape(-1, 142560, 1)

# Função fornecedora para representatividade de faixa
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]

# Carregar modelo Keras
print("📦 Carregando modelo original...")
modelo = tf.keras.models.load_model(arquivo_modelo)

# Converter para TFLite com float16
print("🧪 Iniciando conversão para TFLite (float16)...")
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_types = [tf.float16]
modelo_tflite = converter.convert()

# Salvar modelo quantizado
arquivo_saida = os.path.join(saida, "modelo_quant.tflite")
with open(arquivo_saida, "wb") as f:
    f.write(modelo_tflite)

print(f"✅ Modelo quantizado salvo em: {arquivo_saida}")
