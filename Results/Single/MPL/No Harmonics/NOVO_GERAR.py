# NOVO_GERAR.py  (REGERAR_GRAFICOS)
# Replota curvas de treino/val SEM retreinar, SEM sobreposição visual.

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import tensorflow as tf

# ====== CONFIG ======
PREFER_SUBFOLDER = "modelo_final_quantizado"   # pasta padrão dos artefatos

use_recomputed_history = False
recompute_from_checkpoints = False

# Aparência (apenas visual; NÃO altera métricas/época)
x_jitter = 0.45                 # desloca o TRAIN à esquerda (época)
gap_frac = 0.02                 # 2% do range vertical como vão mínimo
gap_abs  = 0.08                 # ou 0.08 kWh, o que for MAIOR

# Halos (contornos brancos)
halo_train = 0.0                # << azul SEM halo (linha "normal")
halo_val   = 6.0                # laranja com halo p/ destacar

# Cores
COLOR_TRAIN = "#1f77b4"  # azul
COLOR_VAL   = "#ff7f0e"  # laranja

# ====== Auto-detecção do diretório de artefatos ======
CWD = os.getcwd()
candidate_a = os.path.join(CWD, PREFER_SUBFOLDER)  # quando roda de cima
candidate_b = CWD                                  # quando roda de dentro

def _has_history(folder):
    return os.path.exists(os.path.join(folder, "training_history.json")) or \
           os.path.exists(os.path.join(folder, "history_recomputed.json"))

if _has_history(candidate_a):
    quant_folder = candidate_a
elif _has_history(candidate_b):
    quant_folder = candidate_b
else:
    quant_folder = candidate_a if os.path.isdir(candidate_a) else candidate_b

print(f"🔎 usando quant_folder = {quant_folder}")

# ====== Paths ======
hist_json_path = os.path.join(quant_folder, "training_history.json")
hist_recomp    = os.path.join(quant_folder, "history_recomputed.json")
ckpt_dir       = os.path.join(quant_folder, "checkpoints")
arch_json      = os.path.join(quant_folder, "model_arch.json")
X_train_path   = os.path.join(quant_folder, "X_train.npy")
y_train_path   = os.path.join(quant_folder, "y_train.npy")
X_val_path     = os.path.join(quant_folder, "X_val.npy")
y_val_path     = os.path.join(quant_folder, "y_val.npy")

# ====== Util ======
def load_history():
    path = hist_recomp if (use_recomputed_history and os.path.exists(hist_recomp)) else hist_json_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Histórico não encontrado: {path}")
    with open(path, "r") as f:
        return json.load(f)

def ensure_model():
    if not os.path.exists(arch_json):
        raise FileNotFoundError(f"Arquivo de arquitetura não encontrado: {arch_json}")
    with open(arch_json, "r") as f:
        model = tf.keras.models.model_from_json(f.read())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mae",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model

def recompute_all():
    hist = load_history()
    n_epochs = max(len(hist.get(k, [])) for k in [
        "loss","mae","rmse","val_loss","val_mae","val_rmse","eval_loss","eval_mae","eval_rmse"
    ])
    if n_epochs == 0:
        raise ValueError("Histórico sem épocas.")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Pasta checkpoints/ não encontrada: {ckpt_dir}")

    for p in [X_train_path, y_train_path, X_val_path, y_val_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Arquivo necessário não encontrado: {p}")

    Xtr, ytr = np.load(X_train_path), np.load(y_train_path)
    Xv,  yv  = np.load(X_val_path),   np.load(y_val_path)
    model = ensure_model()

    keys = ["loss","mae","rmse","val_loss","val_mae","val_rmse","eval_loss","eval_mae","eval_rmse"]
    out = {k: [] for k in keys}

    for e in range(1, n_epochs + 1):
        ckpt = os.path.join(ckpt_dir, f"epoch_{e:03d}.weights.h5")
        prev = {k: hist.get(k, []) for k in keys}
        if not os.path.exists(ckpt):
            for k in keys:
                out[k].append(prev[k][e-1] if len(prev[k]) >= e else float("nan"))
            continue
        model.load_weights(ckpt)
        ev_tr = model.evaluate(Xtr, ytr, verbose=0)
        ev_va = model.evaluate(Xv,  yv,  verbose=0)
        out["eval_loss"].append(float(ev_tr[0])); out["eval_mae"].append(float(ev_tr[1])); out["eval_rmse"].append(float(ev_tr[2]))
        out["val_loss"].append(float(ev_va[0]));  out["val_mae"].append(float(ev_va[1]));  out["val_rmse"].append(float(ev_va[2]))
        for k in ["loss","mae","rmse"]:
            out[k].append(prev[k][e-1] if len(prev[k]) >= e else float("nan"))

    with open(hist_recomp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✅ history_recomputed.json salvo em {hist_recomp}")
    return out

def val_best_epoch(y):
    i = int(np.nanargmin(y))
    return i + 1, float(y[i])

def forced_separation(y_tr, y_va):
    """
    Cria SEMPRE uma versão visual de validação separada de train por pelo menos 'eps'.
    Não altera 'y_va' real (usado para a melhor época/legendas); só o que é desenhado.
    """
    y_tr = np.asarray(y_tr, float)
    y_va = np.asarray(y_va, float)
    y_all = np.concatenate([y_tr, y_va])
    r = float(np.nanmax(y_all) - np.nanmin(y_all)) or 1.0
    eps = max(gap_abs, gap_frac * r)
    y_plot = y_va.copy()
    need = y_plot <= (y_tr + eps)
    y_plot[need] = y_tr[need] + eps
    return y_plot, eps

def save_plot_csv(base_name, epochs, y_tr, y_va, y_va_plot):
    import pandas as pd
    df = pd.DataFrame({
        "epoch": epochs,
        "train": y_tr,
        "val": y_va,
        "val_plot": y_va_plot,
        "xtrain": epochs - x_jitter,
        "xval": epochs
    })
    out_csv = os.path.join(quant_folder, f"{base_name}_plotdata.csv")
    df.to_csv(out_csv, index=False)
    print(f"💾 CSV do plot: {out_csv}")

def plot_pair(epochs, y_tr, y_va, ylab, out_base, unit="kWh"):
    plt.figure(figsize=(10.5, 7.5))
    ax = plt.gca()

    # X com jitter p/ TRAIN
    x_tr = epochs - x_jitter
    x_va = epochs

    # Curva VAL (visual) sempre separada pelo menos 'eps'
    y_va_plot, eps = forced_separation(y_tr, y_va)

    # ===== TRAIN (azul) contínua, sem tracejado/sem marcadores/sem halo =====
    lw_tr = 2.2
    line_tr, = ax.plot(
        x_tr, y_tr,
        label=f"Train {ylab}",
        color=COLOR_TRAIN,
        lw=lw_tr, alpha=0.98, zorder=2
    )
    if halo_train > 0:
        line_tr.set_path_effects([
            pe.Stroke(linewidth=lw_tr + halo_train, foreground="white"),
            pe.Normal()
        ])

    # ===== VAL (laranja) por cima, com halo =====
    lw_val = 2.5
    line_val, = ax.plot(
        x_va, y_va_plot,
        label=f"Val {ylab}",
        color=COLOR_VAL,
        lw=lw_val, alpha=0.98, zorder=3
    )
    if halo_val > 0:
        line_val.set_path_effects([
            pe.Stroke(linewidth=lw_val + halo_val, foreground="white"),
            pe.Normal()
        ])

    # Faixa entre train e a curva visual de val
    ax.fill_between(epochs, y_tr, y_va_plot, alpha=0.10, zorder=1, color=COLOR_VAL)

    # Melhor época pela validação REAL
    be, bv = val_best_epoch(y_va)
    ax.axvline(be, ls="--", lw=1.2, color="gray", alpha=0.85)
    yb = y_va_plot[be-1]
    x_anno = max(3, int(be * 0.80))
    ax.annotate(
        f"best val = {bv:.3f} {unit} at epoch {be}",
        xy=(be, yb),
        xytext=(x_anno, yb * 1.25 if yb > 0 else yb + 1.0),
        arrowprops=dict(arrowstyle="->", lw=1.1, color="gray", alpha=0.9),
        fontsize=10
    )

    # Eixo X estilo paper + limites cobrindo jitter
    last = int(epochs[-1])
    xticks = [1, 28, 55, 82, 109, 136]
    if last not in xticks:
        xticks.append(last)
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, last + 0.5)

    ax.set_title(f"Training & Validation {ylab}")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.28); ax.legend()
    plt.tight_layout()

    out_png = os.path.join(quant_folder, f"{out_base}.png")
    out_pdf = os.path.join(quant_folder, f"{out_base}.pdf")
    plt.savefig(out_png, dpi=600); plt.savefig(out_pdf)
    plt.close()
    print(f"🖼️  figuras salvas: {out_png} | {out_pdf}")
    save_plot_csv(out_base, epochs, y_tr, y_va, y_va_plot)

# ====== Exec ======
if __name__ == "__main__":
    history = recompute_all() if recompute_from_checkpoints else load_history()

    def pick_pair(metric_key):
        tr = history.get(f"eval_{metric_key}", history.get(metric_key, None))
        va = history.get(f"val_{metric_key}", None)
        if tr is None or va is None:
            print(f"⚠️ Métrica ausente: {metric_key}. Pulando.")
            return None
        tr = np.asarray(tr, float); va = np.asarray(va, float)
        n = min(len(tr), len(va))
        if n == 0:
            print(f"⚠️ Sem pontos para {metric_key}.")
            return None
        tr, va = tr[:n], va[:n]
        epochs = np.arange(1, n+1, dtype=float)
        return epochs, tr, va

    for mkey, ylab, base, unit in [
        ("mae",  "MAE (kWh)",        "training_validation_MAE_kwh",  "kWh"),
        ("rmse", "RMSE (kWh)",       "training_validation_RMSE_kwh", "kWh"),
        ("loss", "Loss (MAE, kWh)",  "training_validation_LOSS_kwh", "kWh"),
    ]:
        pack = pick_pair(mkey)
        if pack is None: 
            continue
        epochs, tr, va = pack
        plot_pair(epochs, tr, va, ylab, base, unit)

    print(f"\n✅ Gráficos gerados em: {os.path.abspath(quant_folder)}")
