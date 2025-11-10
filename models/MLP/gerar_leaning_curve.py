# ==== LEARNING CURVE SCRIPT (MULTI-RUN CORRIGIDO) ====

import os
import numpy as np
import glob
import pickle
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
import math
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

# Seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Params
N_RUNS = 10   # <<< Número de rodadas (você pode mudar)
fractions = np.arange(0.1, 1.01, 0.1)

# Folders
pasta_csv = "amostras_temporarias"
pasta_saida = "learning_curve"
os.makedirs(pasta_saida, exist_ok=True)

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

# Shuffle before PCA
X_full, y_full = shuffle(X_full, y_full, random_state=42)

print(f"\n✅ X_full shape: {X_full.shape}")
print(f"✅ y_full shape: {y_full.shape}")

# Incremental PCA
pca = IncrementalPCA(n_components=min(50, X_full.shape[1]), batch_size=128)

print("\n🚀 Fitting Incremental PCA...")

batch_size_pca = 128
for i in tqdm(range(0, X_full.shape[0], batch_size_pca), desc="PCA batches"):
    batch = X_full[i:i+batch_size_pca]
    pca.partial_fit(batch)

# Transform PCA in batches
X_pca_batches = []
for i in tqdm(range(0, X_full.shape[0], batch_size_pca), desc="Transform batches"):
    batch = X_full[i:i+batch_size_pca]
    X_pca_batch = pca.transform(batch)
    X_pca_batches.append(X_pca_batch)

X_pca = np.vstack(X_pca_batches)

# Learning curve — MULTI RUN
print(f"\n🚀 Starting Learning Curve... ({N_RUNS} runs per fraction)")

mae_matrix = []

for run in range(N_RUNS):
    print(f"\n=== RUN {run+1}/{N_RUNS} ===")
    mae_run = []

    # Shuffle for each run
    X_pca_run, y_run = shuffle(X_pca, y_full, random_state=42 + run)

    for frac in fractions:
        n_samples = int(frac * X_pca_run.shape[0])
        X_train = X_pca_run[:n_samples]
        y_train = y_run[:n_samples]

        print(f"➡️ Training with {n_samples} samples ({frac:.0%})...")

        model = Sequential([
            Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mae", metrics=["mae"])

        # Train
        model.fit(
            X_train, y_train,
            epochs=400,
            batch_size=64,
            validation_split=0.2,
            verbose=0
        )

        # Evaluate MAE
        y_pred = model.predict(X_train).flatten()
        mae = mean_absolute_error(y_train, y_pred)
        mae_run.append(mae)

        print(f"✅ MAE: {mae:.2f}")

    mae_matrix.append(mae_run)

# Convert to array
mae_matrix = np.array(mae_matrix)  # shape (N_RUNS, len(fractions))

# Compute mean and std
mae_mean = np.mean(mae_matrix, axis=0)
mae_std  = np.std(mae_matrix, axis=0)

# Save curve with band
plt.figure(figsize=(8, 6))
plt.plot(fractions * 100, mae_mean, marker="o", linestyle="-", color="b", label="Mean MAE")
plt.fill_between(fractions * 100, mae_mean - mae_std, mae_mean + mae_std, color="blue", alpha=0.2, label="±1 std")
plt.xlabel("Training Set Size (%)")
plt.ylabel("MAE (kWh)")
plt.title(f"Learning Curve (MAE vs Training Size)\n({N_RUNS} runs)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_saida}/learning_curve_mae_vs_size_multi.png")
plt.close()

# Save raw data
df_curve = pd.DataFrame({
    "fraction_percent": fractions * 100,
    "mae_mean": mae_mean,
    "mae_std": mae_std
})
df_curve.to_csv(f"{pasta_saida}/learning_curve_data_multi.csv", index=False)

# Done
print("\n✅ Learning curve (multi-run) completed!")
print(f"📈 Saved: {pasta_saida}/learning_curve_mae_vs_size_multi.png")
print(f"📄 Saved raw data: {pasta_saida}/learning_curve_data_multi.csv")
