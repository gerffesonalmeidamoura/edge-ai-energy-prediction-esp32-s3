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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Seeds
np.random.seed(42)
tf.random.set_seed(42)

# Pastas
data_folder = "amostras_temporarias"
quant_folder = "modelo_final_quantizado"
os.makedirs(quant_folder, exist_ok=True)

# Parâmetros
n_components = 50
batch_size_pca = 64
batch_size_train = 64

# Coleta de arquivos e shuffle
arquivos_csv = sorted(glob.glob(f"{data_folder}/*.csv"))
arquivos_csv, _ = shuffle(arquivos_csv, arquivos_csv, random_state=42)

# PCA incremental
print(f"\n🔍 {len(arquivos_csv)} arquivos encontrados.")
print("🚀 Ajustando PCA incrementalmente...")
pca = IncrementalPCA(n_components=n_components)

for i in tqdm(range(0, len(arquivos_csv), batch_size_pca), desc="PCA Fit"):
    batch = []
    for path in arquivos_csv[i:i+batch_size_pca]:
        arr = np.nan_to_num(np.loadtxt(path, delimiter=",", skiprows=1), nan=0.0)
        batch.append(arr.flatten())
    batch_arr = np.vstack(batch).astype("float32")
    pca.partial_fit(batch_arr)

# Transformacao PCA
print("\n🔍 Transformando dados com PCA...")
X_pca = []
y_full = []

for path in tqdm(arquivos_csv, desc="PCA Transform"):
    arr = np.nan_to_num(np.loadtxt(path, delimiter=",", skiprows=1), nan=0.0)
    X_pca.append(pca.transform(arr.flatten().reshape(1, -1))[0])
    y_full.append(float(os.path.basename(path).split("_")[-1].replace(".csv", "")))

X_pca = np.array(X_pca, dtype="float32")
y_full = np.array(y_full, dtype="float32")

print(f"✅ X_pca shape: {X_pca.shape}")
print(f"✅ y_full shape: {y_full.shape}")

# 📂 Salvar arquivos do PCA
with open(os.path.join(quant_folder, "pca.pkl"), "wb") as f:
    pickle.dump(pca, f)
np.save(os.path.join(quant_folder, "X_pca_final.npy"), X_pca)
np.save(os.path.join(quant_folder, "y_final.npy"), y_full)

# Modelo com BatchNorm e callbacks
model = Sequential([
    Dense(128, activation="relu", input_shape=(n_components,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=Adam(1e-3), loss="mae", metrics=["mae"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10)
]

# Treinamento
print("\n🏋️ Treinando modelo...")
hist = model.fit(
    X_pca, y_full,
    validation_split=0.2,
    epochs=400,
    batch_size=batch_size_train,
    shuffle=True,
    callbacks=callbacks,
    verbose=2
)

# Salvar modelo original
model.save(os.path.join(quant_folder, "modelo_final.keras"))

# Quantização seletiva
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer, quantize_apply

print("\n🔧 Anotando camadas Dense para quantização...")

def quantize_if_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return quantize_annotate_layer(layer)
    return layer

annotated_model = tf.keras.models.clone_model(
    model,
    clone_function=quantize_if_dense
)

quantized_model = quantize_apply(annotated_model)
quantized_model.compile(optimizer=Adam(1e-3), loss="mae", metrics=["mae"])
quantized_model.save(os.path.join(quant_folder, "modelo_final_quantizado.keras"))

# Dataset representativo
def representative_dataset():
    for i in range(min(200, len(X_pca))):
        input_data = np.expand_dims(X_pca[i].astype(np.float32), axis=0)
        yield [input_data]

# Converter para TFLite
print("\n📦 Convertendo para TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

with open(os.path.join(quant_folder, "modelo_quant.tflite"), "wb") as f:
    f.write(tflite_model)

# Avaliação
y_pred = model.predict(X_pca).flatten()
mae = mean_absolute_error(y_full, y_pred)
rmse = math.sqrt(mean_squared_error(y_full, y_pred))
r2 = r2_score(y_full, y_pred)

with open(os.path.join(quant_folder, "metricas.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f} kWh\n")
    f.write(f"RMSE: {rmse:.2f} kWh\n")
    f.write(f"R2: {r2:.4f}\n")

# Histórico serializável
history_clean = {k: [float(x) for x in v] for k, v in hist.history.items()}
with open(os.path.join(quant_folder, "training_history.json"), "w") as f:
    json.dump(history_clean, f, indent=2)

# Função de plotagem dupla (MAE e Loss)
def plot_training_curves(history, folder):
    epochs = range(1, len(history["loss"]) + 1)

    # MAE
    plt.figure(figsize=(8,6))
    plt.plot(epochs, history["mae"], label="Train MAE (kWh)")
    plt.plot(epochs, history["val_mae"], label="Val MAE (kWh)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (kWh)")
    plt.title("Training & Validation MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "training_validation_mae_kwh.png"))
    plt.close()

    # Loss (igual ao MAE)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, history["loss"], label="Train Loss (kWh)")
    plt.plot(epochs, history["val_loss"], label="Val Loss (kWh)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (kWh)")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "training_validation_loss_kwh.png"))
    plt.close()

plot_training_curves(history_clean, quant_folder)

print("\n✅ Tudo pronto! Modelos, métricas e gráficos salvos.")
