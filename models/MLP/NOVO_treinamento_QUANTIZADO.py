import os
import glob
import pickle
import json
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─── Seeds ─────────────────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ─── Folders ───────────────────────────────────────────────────────────────────
data_folder      = "amostras_3dias_sem_harmonicas"
output_folder    = "modelo_quantizado"
os.makedirs(output_folder, exist_ok=True)

# ─── Parameters ────────────────────────────────────────────────────────────────
sample_step      = 10       # downsampling factor (igual ao usado antes)
n_components     = 50       # PCA components
batch_size_pca   = 128      # batches for IncrementalPCA
batch_size_train = 64       # batches for training
epochs           = 400      # training epochs

# ─── 1) Load & Preprocess CSVs ────────────────────────────────────────────────
csv_paths = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
X_list, y_list = [], []
print("\n🔍 Reading and preprocessing CSVs...")
for path in tqdm(csv_paths, desc="Loading CSVs"):
    df = pd.read_csv(path)
    df = df.drop(df.columns[0], axis=1)                            # remove coluna de tempo
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]          # descarta colunas vazias
    df = df.fillna(0.0)
    df = df.iloc[::sample_step, :].reset_index(drop=True)         # downsample
    arr = df.to_numpy(dtype="float32").flatten()
    label = float(os.path.basename(path).split("_")[-1].replace(".csv", ""))
    X_list.append(arr)
    y_list.append(label)

X_full = np.stack(X_list)
y_full = np.array(y_list, dtype="float32")
X_full, y_full = shuffle(X_full, y_full, random_state=42)

print(f"\n✅ X_full shape: {X_full.shape}")
print(f"✅ y_full shape: {y_full.shape}")

# ─── 2) Incremental PCA ────────────────────────────────────────────────────────
pca = IncrementalPCA(n_components=n_components)
print(f"\n🚀 Fitting IncrementalPCA in batches of {batch_size_pca}...")

for i in tqdm(range(0, X_full.shape[0], batch_size_pca), desc="PCA Fit"):
    batch = X_full[i : i + batch_size_pca]
    pca.partial_fit(batch)

print("🚀 Transforming all data with PCA...")
X_pca = pca.transform(X_full).astype("float32")

# Save PCA and transformed data
with open(os.path.join(output_folder, "pca.pkl"), "wb") as f:
    pickle.dump(pca, f)
np.save(os.path.join(output_folder, "X_pca_final.npy"), X_pca)
np.save(os.path.join(output_folder, "y_final.npy"), y_full)

# ─── 3) Build & Train MLP ─────────────────────────────────────────────────────
model = Sequential([
    Dense(128, activation="relu", input_shape=(n_components,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer=Adam(1e-3), loss="mae", metrics=["mae"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10)
]

print("\n🏋️ Training MLP on PCA features...")
hist = model.fit(
    X_pca, y_full,
    epochs=epochs,
    batch_size=batch_size_train,
    validation_split=0.2,
    shuffle=True,
    callbacks=callbacks,
    verbose=2
)

# Save Keras model
model.save(os.path.join(output_folder, "modelo_quantizado.keras"))

# ─── 4) Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_pca).flatten()
mae   = mean_absolute_error(y_full, y_pred)
rmse  = math.sqrt(mean_squared_error(y_full, y_pred))
r2    = r2_score(y_full, y_pred)

with open(os.path.join(output_folder, "metricas.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.4f}\n")

history_clean = {k: [float(vv) for vv in vv_list] for k, vv_list in hist.history.items()}
with open(os.path.join(output_folder, "training_history.json"), "w") as f:
    json.dump(history_clean, f, indent=2)

# Plot training/validation loss and MAE
epochs_range = range(1, len(history_clean["loss"]) + 1)
plt.figure(figsize=(8,6))
plt.plot(epochs_range, history_clean["loss"], label="Train MAE")
plt.plot(epochs_range, history_clean["val_loss"], label="Val MAE")
plt.xlabel("Epoch"); plt.ylabel("MAE (kWh)")
plt.title("Training & Validation MAE")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(output_folder, "training_validation_mae_kwh.png"))
plt.close()

# ─── 5) Convert to TFLite (int8) ──────────────────────────────────────────────
print("\n🚀 Converting to TFLite with INT8 quantization...")

# Wrap sequential model as functional for converter
inp = Input(shape=(n_components,), dtype=tf.float32, name="pca_input")
out = model(inp)
func_model = Model(inputs=inp, outputs=out)

def representative_dataset():
    for i in range(0, X_pca.shape[0], 100):
        yield [X_pca[i : i+1].astype("float32")]

converter = tf.lite.TFLiteConverter.from_keras_model(func_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops   = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type        = tf.int8
converter.inference_output_type       = tf.int8

tflite_model = converter.convert()
with open(os.path.join(output_folder, "modelo_quantizado.tflite"), "wb") as f:
    f.write(tflite_model)

print(f"\n✅ All artifacts saved under '{output_folder}/':\n"
      f"  • Keras model:        modelo_quantizado.keras\n"
      f"  • PCA object:         pca.pkl\n"
      f"  • PCA data arrays:    X_pca_final.npy, y_final.npy\n"
      f"  • Training metrics:   metricas.txt\n"
      f"  • Training history:   training_history.json\n"
      f"  • Loss plot:          training_validation_mae_kwh.png\n"
      f"  • Quantized TFLite:   modelo_quantizado.tflite\n\n"
      "🎉 Training + quantization complete!")
