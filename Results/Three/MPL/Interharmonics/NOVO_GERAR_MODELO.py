# NOVO_Gerar.py
# Avalia o modelo salvo (.tflite preferido; .keras com QuantizeLayer via quantize_scope)
# Gera métricas (MAE, RMSE, R2) e CSV de predições para plot posterior.

import os
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

# Se quiser garantir CPU (evitar erros de GPU), deixe True:
FORCE_CPU = True
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import math
import numpy as np
from pathlib import Path
import tensorflow as tf

BASE = Path(__file__).resolve().parent

# Artefatos esperados
X_PATH      = BASE / "X_pca_final.npy"
Y_PATH      = BASE / "y_final.npy"
MODEL_TFLITE= BASE / "modelo_final_quantizado.tflite"
MODEL_KERAS = BASE / "modelo_final_quantizado.keras"  # ou "modelo_final.keras"

OUT_DIR     = BASE / "artifacts_eval"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------- util ----------
def predict_with_tflite(model_path: Path, X: np.ndarray) -> np.ndarray:
    inter = tf.lite.Interpreter(model_path=str(model_path))
    inter.allocate_tensors()
    inp = inter.get_input_details()[0]
    out = inter.get_output_details()[0]
    ypred = np.empty((X.shape[0],), dtype=np.float32)
    for i in range(X.shape[0]):
        xi = X[i].reshape(1, -1).astype(np.float32)
        inter.set_tensor(inp["index"], xi)
        inter.invoke()
        ypred[i] = float(inter.get_tensor(out["index"]).reshape(-1)[0])
    return ypred

def predict_with_keras(model_path: Path, X: np.ndarray) -> np.ndarray:
    # tenta carregar com quantize_scope (TFMOT)
    try:
        import tensorflow_model_optimization as tfmot
        with tfmot.quantization.keras.quantize_scope():
            model = tf.keras.models.load_model(model_path, compile=False)
    except ImportError:
        raise RuntimeError(
            "O modelo .keras contém camadas de quantização (QuantizeLayer) e requer "
            "`tensorflow-model-optimization`. Instale com:\n\n"
            "    pip install tensorflow-model-optimization\n"
        )
    except ValueError as e:
        # Se der Unknown layer mesmo com quantize_scope, detalhe o erro
        raise RuntimeError(
            f"Falha ao carregar {model_path.name} com quantize_scope: {e}\n"
            "Garanta que a versão do tensorflow_model_optimization seja compatível."
        )
    X2 = X.reshape((X.shape[0], -1)).astype(np.float32)
    ypred = model.predict(X2, batch_size=256, verbose=0).astype(np.float32).reshape(-1)
    return ypred

# ---------- checagens ----------
if not X_PATH.exists() or not Y_PATH.exists():
    raise FileNotFoundError("X_pca_final.npy e/ou y_final.npy não encontrados nesta pasta.")

X = np.load(X_PATH).astype(np.float32)
y = np.load(Y_PATH).astype(np.float32).reshape(-1)
if X.shape[0] != y.shape[0]:
    raise ValueError(f"N exemplos difere: X={X.shape[0]} vs y={y.shape[0]}.")

# ---------- escolha do modelo ----------
# Preferimos TFLite (não depende do TFMOT); se não houver, tentamos .keras com quantize_scope
MODEL_PATH = None
if MODEL_TFLITE.exists():
    MODEL_PATH = MODEL_TFLITE
elif MODEL_KERAS.exists():
    MODEL_PATH = MODEL_KERAS
else:
    # tenta qualquer .tflite ou .keras na pasta
    tfs = sorted(BASE.glob("*.tflite"))
    krs = sorted(BASE.glob("*.keras"))
    if tfs:
        MODEL_PATH = tfs[0]
    elif krs:
        MODEL_PATH = krs[0]
    else:
        raise FileNotFoundError("Nenhum modelo .tflite ou .keras encontrado.")

print(f"Usando modelo: {MODEL_PATH.name}")

# ---------- predições ----------
if MODEL_PATH.suffix == ".tflite":
    y_hat = predict_with_tflite(MODEL_PATH, X)
else:
    y_hat = predict_with_keras(MODEL_PATH, X)

# ---------- métricas ----------
err  = y - y_hat
mae  = float(np.mean(np.abs(err)))
rmse = float(np.sqrt(np.mean(err**2)))
ss_res = float(np.sum(err**2))
ss_tot = float(np.sum((y - np.mean(y))**2))
r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float("nan")

# ---------- salvar ----------
(OUT_DIR / "metrics_from_artifacts.json").write_text(
    json.dumps({
        "model": MODEL_PATH.name,
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]) if X.ndim == 2 else int(np.prod(X.shape[1:])),
        "mae": mae, "rmse": rmse, "r2": r2
    }, indent=2),
    encoding="utf-8"
)

with open(OUT_DIR / "predictions_from_artifacts.csv", "w", encoding="utf-8") as f:
    f.write("idx,ground_truth,prediction,error\n")
    for i, (yt, yp) in enumerate(zip(y, y_hat)):
        f.write(f"{i},{float(yt):.6f},{float(yp):.6f},{float(yt-yp):.6f}\n")

print("\n✅ Avaliação concluída:")
print(f"   MAE = {mae:.3f} kWh  |  RMSE = {rmse:.3f} kWh  |  R² = {r2:.4f}")
print("   →", (OUT_DIR / "metrics_from_artifacts.json").resolve())
print("   →", (OUT_DIR / "predictions_from_artifacts.csv").resolve())
