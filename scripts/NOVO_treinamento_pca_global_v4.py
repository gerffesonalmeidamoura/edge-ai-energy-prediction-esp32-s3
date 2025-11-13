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

# Seeds
np.random.seed(42)
tf.random.set_seed(42)

# Folders
data_folder = "dados_teste_3_dias"
output_folder = "modelo_final"
os.makedirs(output_folder, exist_ok=True)

# Parameters
n_components = 50
batch_size_pca = 64
batch_size_train = 64

# Files
arquivos_csv = sorted(glob.glob(f"{data_folder}/*.csv"))
arquivos_csv, _ = shuffle(arquivos_csv, arquivos_csv, random_state=42)

print(f"\n📂 {len(arquivos_csv)} files found.")
print("⚙️ Adjusting IncrementalPCA in batches...")

# Fit PCA incrementally
pca = IncrementalPCA(n_components=n_components)

for i in tqdm(range(0, len(arquivos_csv), batch_size_pca), desc="PCA Fit"):
    batch = []
    for path in arquivos_csv[i:i+batch_size_pca]:
        arr = np.nan_to_num(np.loadtxt(path, delimiter=","), nan=0.0)
        batch.append(arr.flatten())
    batch_arr = np.vstack(batch).astype("float32")
    if batch_arr.shape[0] >= n_components:
        pca.partial_fit(batch_arr)

# Transform
print("\n📊 Applying PCA transform...")
X_pca = []
y = []

for path in tqdm(arquivos_csv, desc="PCA Transform"):
    arr = np.nan_to_num(np.loadtxt(path, delimiter=","), nan=0.0)
    X_pca.append(pca.transform(arr.flatten().reshape(1, -1))[0])
    y.append(float(os.path.basename(path).split("_")[-1].replace(".csv", "")))

X_pca = np.array(X_pca, dtype="float32")
y = np.array(y, dtype="float32")

# Save PCA
with open(os.path.join(output_folder, "pca.pkl"), "wb") as f:
    pickle.dump(pca, f)

# Model
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

# Train
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

# Predict
y_pred = model.predict(X_pca).flatten()

# Metrics
mae = mean_absolute_error(y, y_pred)
rmse = math.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

with open(os.path.join(output_folder, "metricas.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f} kWh\n")
    f.write(f"RMSE: {rmse:.2f} kWh\n")
    f.write(f"R2: {r2:.4f}\n")

# Save model and training data
with open(os.path.join(output_folder, "modelo.pkl"), "wb") as f:
    pickle.dump(model, f)
np.save(os.path.join(output_folder, "X_pca_final.npy"), X_pca)
np.save(os.path.join(output_folder, "y_final.npy"), y)

# Save history
history_clean = {k: [float(x) for x in v] for k, v in hist.history.items()}
with open(os.path.join(output_folder, "training_history.json"), "w") as f:
    json.dump(history_clean, f, indent=2)

# Plot training/validation loss (MAE in kWh)
print("\n📈 Generating training/validation loss graph...")
epochs = range(1, len(history_clean["loss"]) + 1)

plt.figure(figsize=(8,6))
plt.plot(epochs, history_clean["loss"], label="Train MAE (kWh)")
plt.plot(epochs, history_clean["val_loss"], label="Val MAE (kWh)")
plt.xlabel("Epoch")
plt.ylabel("MAE (kWh)")
plt.title("Training & Validation MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "training_validation_mae_kwh.png"))
plt.close()

# Plot values (sorted, no offset)
print("📉 Generating prediction vs real comparison...")
idx_sort = np.argsort(y)
y_sorted = y[idx_sort]
y_pred_sorted = y_pred[idx_sort]

plt.figure(figsize=(10,6))
plt.plot(y_sorted, label="Real", marker='o', linewidth=1)
plt.plot(y_pred_sorted, label="Predicted", marker='x', linewidth=1)
plt.xlabel("Sample (ordered by real value)")
plt.ylabel("Consumption (kWh)")
plt.title("Real vs Predicted Comparison (PCA + MLP)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "comparison_real_predicted.png"))
plt.close()

print(f"\n✅ All files saved in: {output_folder}")
