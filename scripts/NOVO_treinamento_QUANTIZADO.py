# ==== FINAL TRAINING + QUANTIZATION FOR ESP32 ====

import os
import numpy as np
import glob
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle
import math
from tqdm import tqdm
import tensorflow as tf
import json
import matplotlib.pyplot as plt

# Seed
tf.random.set_seed(42)
np.random.seed(42)

# Folders
pasta_csv = "amostras_temporarias"
pasta_quantizado = "modelo_quantizado"
os.makedirs(pasta_quantizado, exist_ok=True)

# Load CSVs
arquivos_csv = sorted(glob.glob(f"{pasta_csv}/*.csv"))
X_full = []
y_full = []

print("\n🔍 Reading CSVs...")

for path in tqdm(arquivos_csv, desc="Processing CSVs"):
    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = arr.flatten()
        label = float(os.path.basename(path).split("_")[-1].replace(".csv", ""))
        X_full.append(arr)
        y_full.append(label)
    except Exception as e:
        print(f"⚠️ Error in {os.path.basename(path)}: {e}")

X_full = np.array(X_full, dtype="float32")
y_full = np.array(y_full, dtype="float32")

# Shuffle BEFORE PCA
X_full, y_full = shuffle(X_full, y_full, random_state=42)

print(f"\n✅ X_full shape: {X_full.shape}")
print(f"✅ y_full shape: {y_full.shape}")

# Incremental PCA
pca = IncrementalPCA(n_components=min(50, X_full.shape[1]), batch_size=128)

print("\n🚀 Applying Incremental PCA...")

batch_size_pca = 128
for i in tqdm(range(0, X_full.shape[0], batch_size_pca), desc="PCA batches"):
    batch = X_full[i:i+batch_size_pca]
    pca.partial_fit(batch)

X_pca_batches = []
for i in tqdm(range(0, X_full.shape[0], batch_size_pca), desc="Transform batches"):
    batch = X_full[i:i+batch_size_pca]
    X_pca_batch = pca.transform(batch)
    X_pca_batches.append(X_pca_batch)

X_pca = np.vstack(X_pca_batches).astype(np.float32)

# Build model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_pca.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mae", metrics=["mae"])

# Train model
print("\n🚀 Training model...")
hist = model.fit(
    X_pca, y_full,
    epochs=400,
    batch_size=64,
    validation_split=0.2,
    verbose=2
)

# Predict and metrics
y_pred = model.predict(X_pca).flatten()
mae = mean_absolute_error(y_full, y_pred)
rmse = math.sqrt(mean_squared_error(y_full, y_pred))
r2 = r2_score(y_full, y_pred)

print(f"\n🎯 Final model - MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# Save model
model.save(f"{pasta_quantizado}/modelo_quantizado.keras")

# Save PCA
with open(f"{pasta_quantizado}/pca.pkl", "wb") as f:
    pickle.dump(pca, f)

# Save datasets
np.save(f"{pasta_quantizado}/X_pca_final.npy", X_pca)
np.save(f"{pasta_quantizado}/y_final.npy", y_full)

# Save metrics
with open(f"{pasta_quantizado}/metricas.txt", "w") as f:
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.4f}\n")

# Save training history
with open(f"{pasta_quantizado}/training_history.json", "w") as f:
    json.dump(hist.history, f)

# Plot loss
plt.figure(figsize=(8, 6))
plt.plot(hist.history["loss"], label="Train Loss (MAE)")
plt.plot(hist.history["val_loss"], label="Validation Loss (MAE)")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_quantizado}/training_validation_loss.png")
plt.close()

# Plot MAE
if "mae" in hist.history:
    plt.figure(figsize=(8, 6))
    plt.plot(hist.history["mae"], label="Train MAE")
    plt.plot(hist.history["val_mae"], label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training & Validation MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{pasta_quantizado}/training_validation_mae.png")
    plt.close()

# TFLite quantization (Functional model workaround)
print("\n🚀 Converting to TFLite (int8 quantization)...")

from tensorflow.keras import Input, Model

# Convert Sequential to Functional
input_layer = Input(shape=(X_pca.shape[1],), dtype=tf.float32, name="dense_input")
output_layer = model(input_layer)
functional_model = Model(inputs=input_layer, outputs=output_layer)

# Representative dataset
def representative_dataset():
    for i in range(0, X_pca.shape[0], 100):
        yield [X_pca[i:i+1].astype(np.float32)]

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(functional_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

# Save
with open(f"{pasta_quantizado}/modelo_quantizado.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("\n✅ Model, PCA, metrics, history and quantized TFLite saved.")
print(f"📁 Outputs in: {pasta_quantizado}/")
print("\n🎉 Training + quantization complete!")
