import os
import numpy as np
import glob
import pickle
import json
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Directories
data_folder = "dados_teste_3_dias"
model_folder = "modelo_final"
os.makedirs(model_folder, exist_ok=True)

# Collect and shuffle file paths and labels
file_paths = sorted(glob.glob(f"{data_folder}/*.csv"))
labels = [float(os.path.basename(p).split("_")[-1].replace(".csv", "")) for p in file_paths]
file_paths, labels = shuffle(file_paths, labels, random_state=42)

# Determine number of features from the first file
tmp = np.nan_to_num(np.loadtxt(file_paths[0], delimiter=",", dtype="float32"), nan=0.0)
n_features = tmp.flatten().shape[0]

# PCA parameters
dim = min(50, n_features)
chunk_size = 16  # adjust for memory vs speed

# Initialize IncrementalPCA
pca = IncrementalPCA(n_components=dim)

# Initial PCA fit (ensure at least dim samples)
init_samples = min(dim, len(file_paths))
print(f"\n🚀 PCA initial batch with {init_samples} samples (n_components={dim})...")
init_data = []
for path in tqdm(file_paths[:init_samples], desc="Loading init batch"):
    arr = np.nan_to_num(np.loadtxt(path, delimiter=",", dtype="float32"), nan=0.0)
    init_data.append(arr.flatten())
init_batch = np.vstack(init_data).astype("float32")
pca.partial_fit(init_batch)
del init_data, init_batch

# Continue partial_fit in chunks for speed
print(f"\n🚀 Continuing PCA partial_fit in chunks of {chunk_size}...")
for start in tqdm(range(init_samples, len(file_paths), chunk_size), desc="PCA Partial Fit"):
    end = min(start + chunk_size, len(file_paths))
    batch = []
    for p in file_paths[start:end]:
        arr = np.nan_to_num(np.loadtxt(p, delimiter=",", dtype="float32"), nan=0.0)
        batch.append(arr.flatten())
    batch_arr = np.vstack(batch).astype("float32")
    pca.partial_fit(batch_arr)
    del batch, batch_arr

# Transform into PCA space
print("\n🔍 Transforming files with PCA...")
X_pca_list = []
for start in tqdm(range(0, len(file_paths), chunk_size), desc="PCA Transform"):
    end = min(start + chunk_size, len(file_paths))
    batch = []
    for p in file_paths[start:end]:
        arr = np.nan_to_num(np.loadtxt(p, delimiter=",", dtype="float32"), nan=0.0)
        batch.append(arr.flatten())
    arr_batch = np.vstack(batch).astype("float32")
    X_pca_list.append(pca.transform(arr_batch))
    del batch, arr_batch
X_pca = np.vstack(X_pca_list)
y_full = np.array(labels, dtype="float32")

print(f"\n✅ X_pca shape: {X_pca.shape}")
print(f"✅ y_full shape: {y_full.shape}")

# Build model with normalization and callbacks
model = Sequential([
    Dense(128, activation="relu", input_shape=(dim,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mae",
    metrics=["mae"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10)
]

# Train using validation_split and shuffled batches
print("\n🚀 Training model...")
hist = model.fit(
    X_pca, y_full,
    validation_split=0.2,
    batch_size=64,
    epochs=400,
    shuffle=True,
    callbacks=callbacks,
    verbose=2
)

# Evaluate on full dataset
print("\n🔍 Evaluating model on full dataset...")
y_pred = model.predict(X_pca, verbose=0).flatten()
mae = mean_absolute_error(y_full, y_pred)
rmse = math.sqrt(mean_squared_error(y_full, y_pred))
r2 = r2_score(y_full, y_pred)
print(f"\n🎯 Final model - MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# Save artifacts
model.save(os.path.join(model_folder, "modelo_final.keras"))
with open(os.path.join(model_folder, "pca.pkl"), "wb") as f:
    pickle.dump(pca, f)
np.save(os.path.join(model_folder, "X_pca_final.npy"), X_pca)
np.save(os.path.join(model_folder, "y_final.npy"), y_full)

# Convert history to native Python floats before JSON serialization
history_clean = {key: [float(val) for val in values] for key, values in hist.history.items()}
with open(os.path.join(model_folder, "training_history.json"), "w") as f:
    json.dump(history_clean, f, indent=2)

# Save metrics
with open(os.path.join(model_folder, "metricas.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.4f}\n")

# Plotting function
def plot_training_graphs(history, output_folder):
    plt.figure(figsize=(8,6))
    plt.plot(history["loss"], label="Train Loss (MAE)")
    plt.plot(history["val_loss"], label="Validation Loss (MAE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE)")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "training_validation_loss.png"))
    plt.close()

    if "mae" in history and "val_mae" in history:
        plt.figure(figsize=(8,6))
        plt.plot(history["mae"], label="Train MAE")
        plt.plot(history["val_mae"], label="Validation MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Training & Validation MAE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "training_validation_mae.png"))
        plt.close()

plot_training_graphs(history_clean, model_folder)

print("\n✅ Process completed with smooth learning curves and serializable history.")
