import os, json, math, argparse
from pathlib import Path
import numpy as np

# -------------------- CLI --------------------
p = argparse.ArgumentParser(description="Compute RMSE/MAE from saved artifacts and (optionally) estimate epochwise RMSE.")
p.add_argument("--artifacts_dir", default=".", help="Pasta onde estão o modelo e os .npy/.json (default: .)")
p.add_argument("--model", default=None, help="Caminho do modelo .keras OU .tflite (se não passar, o script tenta achar).")
p.add_argument("--x_file", default="X_pca_final.npy", help="Arquivo com features PCA (default: X_pca_final.npy)")
p.add_argument("--y_file", default="y_final.npy", help="Arquivo com targets (default: y_final.npy)")
p.add_argument("--history", default="training_history.json", help="Arquivo de histórico (default: training_history.json)")
p.add_argument("--out_dir", default="artifacts_eval", help="Pasta de saída (default: artifacts_eval)")
p.add_argument("--estimate_factor", default="gaussian", choices=["gaussian","laplace","custom"],
               help="Fator para estimar RMSE a partir de MAE (gaussian=1.2533, laplace=1.4142, custom=--custom_factor).")
p.add_argument("--custom_factor", type=float, default=1.253314, help="Fator customizado (usado se --estimate_factor=custom).")
args = p.parse_args()

ART = Path(args.artifacts_dir)
OUT = Path(args.out_dir); OUT.mkdir(exist_ok=True, parents=True)

# -------------------- localizar artefatos --------------------
def pick_model():
    if args.model:
        return Path(args.model)
    # tenta alguns nomes comuns
    cands = [
        ART/"modelo_final_quantizado.keras",
        ART/"modelo_final.keras",
        ART/"modelo_quant.tflite",
        ART/"modelo_final_quantizado.tflite",
        ART/"modelo_final_quant.tflite",
    ]
    for c in cands:
        if c.exists(): return c
    # varre por qualquer .keras ou .tflite
    ks = sorted(ART.glob("*.keras"))
    ts = sorted(ART.glob("*.tflite"))
    if ks: return ks[0]
    if ts: return ts[0]
    raise FileNotFoundError("Não encontrei um modelo (.keras ou .tflite). Use --model para apontar.")

MODEL_PATH = pick_model()
X_PATH = ART/args.x_file
Y_PATH = ART/args.y_file
HIST_PATH = ART/args.history

for path in [X_PATH, Y_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

# -------------------- carregar dados --------------------
X = np.load(X_PATH).astype(np.float32)
y = np.load(Y_PATH).astype(np.float32).reshape(-1)
if X.shape[0] != y.shape[0]:
    raise ValueError(f"N exemplos de X ({X.shape[0]}) ≠ y ({y.shape[0]}).")

# -------------------- inferência --------------------
def predict_with_keras(model_path, X_):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    # garante dtype/shape (N, n_comp)
    X2 = X_.reshape((X_.shape[0], -1)).astype(np.float32)
    y_pred = model.predict(X2, batch_size=256, verbose=0).astype(np.float32).reshape(-1)
    return y_pred

def predict_with_tflite(model_path, X_):
    import tensorflow as tf
    inter = tf.lite.Interpreter(model_path=str(model_path))
    inter.allocate_tensors()
    inp = inter.get_input_details()[0]
    out = inter.get_output_details()[0]
    y_pred = np.empty((X_.shape[0],), dtype=np.float32)
    # tflite geralmente espera (1, n_comp)
    for i in range(X_.shape[0]):
        xi = X_[i].reshape(1, -1).astype(np.float32)
        inter.set_tensor(inp["index"], xi)
        inter.invoke()
        y_pred[i] = float(inter.get_tensor(out["index"]).reshape(-1)[0])
    return y_pred

if MODEL_PATH.suffix == ".keras":
    y_hat = predict_with_keras(MODEL_PATH, X)
elif MODEL_PATH.suffix == ".tflite":
    y_hat = predict_with_tflite(MODEL_PATH, X)
else:
    raise ValueError(f"Extensão de modelo não suportada: {MODEL_PATH.suffix}")

# -------------------- métricas exatas do modelo final --------------------
err = y - y_hat
mae  = float(np.mean(np.abs(err)))
rmse = float(np.sqrt(np.mean(err**2)))
ss_res = float(np.sum(err**2))
ss_tot = float(np.sum((y - np.mean(y))**2))
r2   = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float("nan")

metrics_json = {
    "model_path": str(MODEL_PATH),
    "n_samples": int(X.shape[0]),
    "feature_dim": int(X.shape[1]) if X.ndim == 2 else int(np.prod(X.shape[1:])),
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
}
(OUT/"metrics_from_artifacts.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

with open(OUT/"predictions_from_artifacts.csv", "w", encoding="utf-8") as f:
    f.write("idx,ground_truth,prediction,error\n")
    for i, (yt, yp) in enumerate(zip(y, y_hat)):
        f.write(f"{i},{float(yt):.6f},{float(yp):.6f},{float(yt-yp):.6f}\n")

print(f"✔ Métricas (EXATAS) salvas em: {OUT/'metrics_from_artifacts.json'}")
print(f"✔ Predições salvas em:         {OUT/'predictions_from_artifacts.csv'}")

# -------------------- estimar RMSE por época (se faltar no histórico) --------------------
if HIST_PATH.exists():
    hist = json.loads(HIST_PATH.read_text(encoding="utf-8"))
    # alguns histories vêm embrulhados em {"history": {...}}
    if "history" in hist and isinstance(hist["history"], dict):
        hist = hist["history"]

    have_rmse = ("rmse" in hist) and ("val_rmse" in hist)
    if have_rmse:
        print("ℹ Histórico já contém rmse/val_rmse; não é necessário estimar.")
    else:
        # define fator
        if args.estimate_factor == "gaussian":
            factor = 1.2533141373155  # RMSE ≈ 1.2533 * MAE para erro ~ Normal(0, σ)
            used = "gaussian"
        elif args.estimate_factor == "laplace":
            factor = 1.4142135623731  # RMSE ≈ √2 * MAE para erro ~ Laplace(0, b)
            used = "laplace"
        else:
            factor = float(args.custom_factor)
            used = "custom"

        mae_series     = np.asarray(hist.get("mae", []), dtype=float)
        val_mae_series = np.asarray(hist.get("val_mae", []), dtype=float)

        rmse_est     = (factor * mae_series).tolist()     if mae_series.size     else []
        val_rmse_est = (factor * val_mae_series).tolist() if val_mae_series.size else []

        hist_est = dict(hist)  # cópia rasa
        hist_est["rmse_est"]     = rmse_est
        hist_est["val_rmse_est"] = val_rmse_est
        out_hist = {
            "history": hist_est,
            "_note": f"RMSE estimado a partir de MAE com fator={factor:.6f} ({used}). "
                     f"Este arquivo foi gerado apenas para plot/analise; não é RMSE real por época."
        }
        (OUT/"training_history_with_rmse_estimated.json").write_text(
            json.dumps(out_hist, indent=2), encoding="utf-8"
        )

        # melhor época segundo val_rmse_est
        best_epoch = None
        best_val = None
        if val_rmse_est:
            best_idx = int(np.argmin(val_rmse_est)) + 1
            best_epoch = best_idx
            best_val = float(val_rmse_est[best_idx-1])

        with open(OUT/"training_best_epochs_estimated.txt", "w", encoding="utf-8") as f:
            f.write(f"factor_used: {used} ({factor:.6f})\n")
            if best_epoch is not None:
                f.write(f"val_rmse_est: epoch {best_epoch}, value {best_val:.6f}\n")
            if len(val_mae_series):
                b_mae = int(np.argmin(val_mae_series)) + 1
                f.write(f"val_mae: epoch {b_mae}, value {float(val_mae_series[b_mae-1]):.6f}\n")

        print("✔ Histórico com RMSE ESTIMADO salvo em:",
              OUT/"training_history_with_rmse_estimated.json")
        print("✔ Melhores épocas (estimado) salvo em:  ",
              OUT/"training_best_epochs_estimated.txt")
else:
    print("⚠ training_history.json não encontrado; pulei a etapa de estimar RMSE por época.")

print("\nResumo:")
print(json.dumps(metrics_json, indent=2))
