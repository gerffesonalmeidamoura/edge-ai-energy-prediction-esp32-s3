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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Folders
data_folder   = "amostras_3dias_sem_harmonicas"
output_folder = "modelo_final"
os.makedirs(output_folder, exist_ok=True)

# Parameters
n_components     = 50
batch_size_pca   = n_components     # must be >= n_components
batch_size_train = 64
sample_step      = 10               # downsampling factor on time axis

# Gather and shuffle files
arquivos_csv = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
arquivos_csv, _ = shuffle(arquivos_csv, arquivos_csv, random_state=42)

print(f"\n📂 {len(arquivos_csv)} files found.")
print(f"⚙️ Adjusting IncrementalPCA in batches of {batch_size_pca}, downsampling every {sample_step} rows...")

# Initialize IncrementalPCA
pca = IncrementalPCA(n_components=n_components)

# --------------------------------
# 1) PCA Fit
# --------------------------------
for i in tqdm(range(0, len(arquivos_csv), batch_size_pca), desc="PCA Fit"):
    batch = []
    for path in arquivos_csv[i : i + batch_size_pca]:
        df = pd.read_csv(path)
        # drop time column and any unnamed empty columns
        df = df.drop(df.columns[0], axis=1)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df.fillna(0.0)
        # downsample on time axis
        df = df.iloc[::sample_step, :].reset_index(drop=True)
        arr = df.to_numpy(dtype="float32")
        batch.append(arr.flatten())
    batch_arr = np.vstack(batch).astype("float32")
    pca.partial_fit(batch_arr)

# --------------------------------
# 2) PCA Transform
# --------------------------------
print("\n📊 Applying PCA transform...")
X_pca = []
y     = []

for path in tqdm(arquivos_csv, desc="PCA Transform"):
    df = pd.read_csv(path)
    df = df.drop(df.columns[0], axis=1)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.fillna(0.0)
    df = df.iloc[::sample_step, :].reset_index(drop=True)
    arr = df.to_numpy(dtype="float32").flatten().reshape(1, -1)
    X_pca.append(pca.transform(arr)[0])
    # extract target (Energy(kWh) ALL) from filename
    y_val = float(os.path.basename(path).split("_")[-1].replace(".csv", ""))
    y.append(y_val)

X_pca = np.array(X_pca, dtype="float32")
y     = np.array(y, dtype="float32")

# Save PCA
with open(os.path.join(output_folder, "pca.pkl"), "wb") as f:
    pickle.dump(pca, f)

# --------------------------------
# 3) Build & train MLP
# --------------------------------
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

print("\n🏋️ Training model...")
hist = model.fit(
    X_pca, y,
    validation_split=0.2,
    epochs=400,
    batch_size=batch_size_train,
    shuffle=True,
    callbacks=callbacks,
    verbose=2
)

# Save model
model.save(os.path.join(output_folder, "modelo_final.keras"))

# --------------------------------
# 4) Predict & Metrics
# --------------------------------
y_pred = model.predict(X_pca).flatten()
mae   = mean_absolute_error(y, y_pred)
rmse  = math.sqrt(mean_squared_error(y, y_pred))
r2    = r2_score(y, y_pred)

with open(os.path.join(output_folder, "metricas.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f} kWh\n")
    f.write(f"RMSE: {rmse:.2f} kWh\n")
    f.write(f"R2: {r2:.4f}\n")

# Save training artifacts
with open(os.path.join(output_folder, "modelo.pkl"), "wb") as f:
    pickle.dump(model, f)
np.save(os.path.join(output_folder, "X_pca_final.npy"), X_pca)
np.save(os.path.join(output_folder, "y_final.npy"), y)

history_clean = {k: [float(x) for x in v] for k, v in hist.history.items()}
with open(os.path.join(output_folder, "training_history.json"), "w") as f:
    json.dump(history_clean, f, indent=2)

# --------------------------------
# 5) Plots
# --------------------------------
print("\n📈 Generating training/validation MAE graph...")
epochs = range(1, len(history_clean["loss"]) + 1)
plt.figure(figsize=(8,6))
plt.plot(epochs, history_clean["loss"], label="Train MAE (kWh)")
plt.plot(epochs, history_clean["val_loss"], label="Val MAE (kWh)")
plt.xlabel("Epoch"); plt.ylabel("MAE (kWh)")
plt.title("Training & Validation MAE")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(output_folder, "training_validation_mae_kwh.png"))
plt.close()

print("📉 Generating prediction vs real comparison...")
idx = np.argsort(y)
plt.figure(figsize=(10,6))
plt.plot(y[idx], label="Real", marker='o', linewidth=1)
plt.plot(y_pred[idx], label="Predicted", marker='x', linewidth=1)
plt.xlabel("Sample (ordered by real value)")
plt.ylabel("Consumption (kWh)")
plt.title("Real vs Predicted Comparison (PCA + MLP)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(output_folder, "comparison_real_predicted.png"))
plt.close()

print(f"\n✅ All files saved in: {output_folder}")
