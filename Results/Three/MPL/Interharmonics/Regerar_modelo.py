# retrain_from_pca.py
# Re-treina um MLP a partir de X_pca_final.npy (features) e y_final.npy (alvos)
# Gera RMSE/MAE por época, gráficos e artefatos para o artigo.

import os, json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------------- Configs ----------------
SEED        = 42
EPOCHS      = 300
BATCH_SIZE  = 64
LR          = 5e-3
VAL_SPLIT   = 0.2
OUT_DIR     = Path("retrain_from_pca_artifacts"); OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR    = OUT_DIR / "plots"; PLOT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- Dados ----------------
X = np.load("X_pca_final.npy").astype("float32")  # shape (N, n_components)
y = np.load("y_final.npy").astype("float32").reshape(-1)

if X.shape[0] != y.shape[0]:
    raise ValueError(f"X e y com comprimentos diferentes: {X.shape[0]} vs {y.shape[0]}")

N = X.shape[0]
idx = np.random.permutation(N)
n_val = int(np.round(N * VAL_SPLIT))
val_idx = idx[:n_val]
trn_idx = idx[n_val:]

X_tr, y_tr = X[trn_idx], y[trn_idx]
X_va, y_va = X[val_idx], y[val_idx]

print(f"X: {X.shape} | y: {y.shape}  -> train={X_tr.shape[0]}  val={X_va.shape[0]}")

# ---------------- Modelo (MLP) ----------------
model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.30),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.30),
    Dense(1)
])

model.compile(
    optimizer=Adam(LR),
    loss="mae",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse")
    ]
)

ckpt_path = OUT_DIR / "best_model.keras"
cbs = [
    EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10),
    ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss",
                    save_best_only=True, save_weights_only=False)
]

hist = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=cbs,
    verbose=2
)

# ---------------- Salvar histórico ----------------
history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
(OUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2))

with open(OUT_DIR / "training_history.csv", "w", newline="") as f:
    w = csv.writer(f)
    keys = list(history.keys())
    w.writerow(["epoch"] + keys)
    for i in range(len(next(iter(history.values())))):
        w.writerow([i+1] + [history[k][i] for k in keys])

# melhores épocas (val_mae e val_rmse, se existirem)
best_lines = []
for k in ("val_mae", "val_rmse", "val_loss"):
    if k in history:
        e = int(np.argmin(history[k])) + 1
        best_lines.append(f"{k}: epoch {e}, value {history[k][e-1]:.6f}")
(OUT_DIR / "training_best_epochs.txt").write_text("\n".join(best_lines))

# ---------------- Gráficos (MAE e RMSE por época) ----------------
def moving_average(x, k=3):
    x = np.asarray(x, float)
    if k > 1 and len(x) >= k:
        return np.convolve(x, np.ones(k)/k, mode="same")
    return x

def nice_limits(y1, y2, pad=0.05, cap_q=0.995):
    yy = np.r_[np.asarray(y1,float), np.asarray(y2,float)]
    lo = float(np.nanmin(yy))
    hi = float(np.nanquantile(yy, cap_q))
    rng = max(1e-9, hi - lo)
    return lo - pad*rng, hi + pad*rng

def plot_pair(ytr, yva, title, ylab, fname, smooth_val=3):
    ytr = np.asarray(ytr, float)
    yva = np.asarray(yva, float)
    n = int(min(len(ytr), len(yva)))
    x = np.arange(1, n+1)
    va_s = moving_average(yva[:n], smooth_val)

    fig, ax = plt.subplots(figsize=(8,6))
    # guardo os handles para controlar a ordem na legenda (Train em cima, Val embaixo)
    line_tr, = ax.plot(x, ytr[:n], label=f"Train {ylab}", lw=1.8)
    line_va, = ax.plot(x, va_s,      label=f"Val {ylab}",   lw=1.8)

    be = int(np.argmin(yva[:n])) + 1
    bv = float(yva[be-1])
    ax.axvline(be, ls="--", lw=1.2, color="gray")

    ylo, yhi = nice_limits(ytr[:n], yva[:n], pad=0.05, cap_q=0.995)
    ax.set_ylim(ylo, yhi)
    yr = yhi - ylo
    anchor = max(ytr[be-1], yva[be-1], va_s[be-1])
    yt = min(yhi - 0.05*yr, anchor + 0.12*yr)
    x_off = max(5, n//12)
    xt, ha = (max(1, be - x_off), "right") if be/n > 0.75 else (min(n, be + x_off), "left")
    ax.annotate(f"best val = {bv:.3f} kWh at epoch {be}",
                xy=(be, bv), xytext=(xt, yt), ha=ha, va="bottom",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="gray"),
                fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylab)
    ax.set_xlim(1, n*1.02)
    step = max(1, int(np.ceil(n/6))); ax.set_xticks(np.arange(1, n+1, step))
    ax.grid(True, ls=":", lw=0.6)

    # ---- LEGENDA VERTICAL (Train em cima, Val embaixo), centrada no topo e um pouco mais baixa ----
    handles = [line_tr, line_va]
    labels  = [h.get_label() for h in handles]
    leg = ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),   # levemente abaixo da borda superior
        ncol=1,
        frameon=False,
        borderaxespad=0.2
    )

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{fname}.png", dpi=600)
    fig.savefig(PLOT_DIR / f"{fname}.pdf")
    plt.close(fig)

if "mae" in history and "val_mae" in history:
    plot_pair(history["mae"], history["val_mae"],
              "Training & Validation MAE (kWh)", "MAE (kWh)",
              "training_validation_MAE_kwh", smooth_val=3)

if "rmse" in history and "val_rmse" in history:
    plot_pair(history["rmse"], history["val_rmse"],
              "Training & Validation RMSE (kWh)", "RMSE (kWh)",
              "training_validation_RMSE_kwh", smooth_val=3)

# ---------------- Predições p/ auditar (train/val) ----------------
# (útil para calcular métricas depois sem re-treinar)
yhat_tr = model.predict(X_tr, batch_size=BATCH_SIZE, verbose=0).ravel()
yhat_va = model.predict(X_va, batch_size=BATCH_SIZE, verbose=0).ravel()

with open(OUT_DIR / "predicoes_train_val.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["split","index","y_true","y_pred"])
    for i,(yt,yp) in enumerate(zip(y_tr, yhat_tr)):
        w.writerow(["train", int(trn_idx[i]), float(yt), float(yp)])
    for i,(yt,yp) in enumerate(zip(y_va, yhat_va)):
        w.writerow(["val", int(val_idx[i]), float(yt), float(yp)])

# ---------------- Salvar modelo final ----------------
model.save(OUT_DIR / "retrained_from_pca.keras")
print("\n✅ Pronto! Artefatos em:", OUT_DIR.resolve())
print("   ├─ training_history.json / .csv / training_best_epochs.txt")
print("   ├─ plots: training_validation_MAE_kwh.(png|pdf), training_validation_RMSE_kwh.(png|pdf)")
print("   ├─ predicoes_train_val.csv")
print("   └─ best_model.keras (checkpoint) e retrained_from_pca.keras")
