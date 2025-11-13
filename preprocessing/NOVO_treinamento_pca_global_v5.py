import os
import numpy as np
import glob
import pickle
import json
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Seeds
tf.random.set_seed(42)
np.random.seed(42)

# Pastas
pasta_csv = "dados_teste_3_dias"
pasta_modelo = "modelo_final"
os.makedirs(pasta_modelo, exist_ok=True)

# Coleta de arquivos
arquivos_csv = sorted(glob.glob(f"{pasta_csv}/*.csv"))
print(f"\n🔍 {len(arquivos_csv)} arquivos encontrados.")

# Embaralhamento
arquivos_csv = shuffle(arquivos_csv, random_state=42)

# Etapa 1: PCA em batches
n_components = 50
batch_size = 64
pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

print("\n🚀 Etapa 1: Ajustando PCA incrementalmente...")

for i in tqdm(range(0, len(arquivos_csv), batch_size), desc="Ajustando PCA"):
    batch_paths = arquivos_csv[i:i+batch_size]
    batch_data = []
    for path in batch_paths:
        try:
            arr = np.loadtxt(path, delimiter=",", dtype="float32")
            if arr.size == 0 or np.isnan(arr).any():
                print(f"⚠️ Ignorado {os.path.basename(path)} (vazio ou contém NaN)")
                continue
            arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
            arr = arr.flatten()
            batch_data.append(arr)
        except Exception as e:
            print(f"⚠️ Erro ao carregar {os.path.basename(path)}: {e}")
            continue
    if batch_data:
        batch_data = np.array(batch_data, dtype="float32")
        if np.isnan(batch_data).any():
            print(f"⚠️ Batch contém NaN — ignorado")
            continue
        if batch_data.shape[0] >= n_components:
            pca.partial_fit(batch_data)
        else:
            print(f"⚠️ Lote com apenas {batch_data.shape[0]} amostras ignorado (mínimo necessário: {n_components})")

# Etapa 2: Transformação PCA + criação do dataset
X_pca = []
y_full = []

print("\n⚙️ Etapa 2: Transformando amostras com PCA...")

for path in tqdm(arquivos_csv, desc="Transformando"):
    try:
        arr = np.loadtxt(path, delimiter=",", dtype="float32")
        if arr.size == 0 or np.isnan(arr).any():
            print(f"⚠️ Ignorado {os.path.basename(path)} (vazio ou contém NaN)")
            continue
        arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
        arr = arr.flatten().reshape(1, -1)
        transformed = pca.transform(arr)
        X_pca.append(transformed[0])
        label = float(os.path.basename(path).split("_")[-1].replace(".csv", ""))
        y_full.append(label)
    except Exception as e:
        print(f"⚠️ Erro em {os.path.basename(path)}: {e}")

X_pca = np.array(X_pca, dtype="float32")
y_full = np.array(y_full, dtype="float32")

print(f"\n✅ X_pca shape: {X_pca.shape}")
print(f"✅ y_full shape: {y_full.shape}")

# Modelo
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_pca.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss="mae", metrics=["mae"])

# Treinamento
print("\n🏋️ Treinando modelo...")
hist = model.fit(
    X_pca, y_full,
    validation_split=0.2,
    epochs=400,
    batch_size=64,
    verbose=2
)

# Predição
y_pred = model.predict(X_pca).flatten()

# Métricas
mae = mean_absolute_error(y_full, y_pred)
rmse = math.sqrt(mean_squared_error(y_full, y_pred))
r2 = r2_score(y_full, y_pred)

print(f"\n🎯 MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# Salvando resultados
model.save(f"{pasta_modelo}/modelo_final.keras")
with open(f"{pasta_modelo}/pca.pkl", "wb") as f:
    pickle.dump(pca, f)
np.save(f"{pasta_modelo}/X_pca_final.npy", X_pca)
np.save(f"{pasta_modelo}/y_final.npy", y_full)

with open(f"{pasta_modelo}/metricas.txt", "w") as f:
    f.write(f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.4f}\n")

with open(f"{pasta_modelo}/training_history.json", "w") as f:
    json.dump(hist.history, f)

# Plots
plt.figure(figsize=(8,6))
plt.plot(hist.history["loss"], label="Train Loss (MAE)")
plt.plot(hist.history["val_loss"], label="Validation Loss (MAE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_modelo}/training_validation_loss.png")
plt.close()

plt.figure(figsize=(8,6))
plt.plot(hist.history["mae"], label="Train MAE")
plt.plot(hist.history["val_mae"], label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("MAE over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_modelo}/training_validation_mae.png")
plt.close()

print("\n✅ Tudo pronto! Modelo e resultados salvos.")
