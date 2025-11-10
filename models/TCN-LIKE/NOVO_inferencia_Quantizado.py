import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import re
import scipy.stats as stats

# === CONFIG ===
PASTA_NPY = r"C:\projeto_artigo\mono\2_TCN\com_interharmonicas\convertidos_down_142560_TESTE"
MODELO_TFLITE = "modelo_mcu_tcn_int8.tflite"
ARQUIVO_SCALER = r"C:\projeto_artigo\mono\2_TCN\com_interharmonicas\scaler_treino.save"
PASTA_SAIDA = os.path.join(os.path.dirname(MODELO_TFLITE), "resultado_inferencia_quantizada")
os.makedirs(PASTA_SAIDA, exist_ok=True)

# === MODEL ===
interpreter = tf.lite.Interpreter(model_path=MODELO_TFLITE)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# === SCALER ===
scaler = joblib.load(ARQUIVO_SCALER)

# === FILES ===
arquivos_npy = sorted([f for f in os.listdir(PASTA_NPY) if f.endswith(".npy")])
y_real, y_pred, nomes = [], [], []

for nome_arq in tqdm(arquivos_npy, desc="🔎 Running inference"):
    caminho = os.path.join(PASTA_NPY, nome_arq)
    arr = np.load(caminho)
    
    if arr.ndim == 1:
        arr = arr.reshape(1, -1, 1)
    elif arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], 1)

    arr = scaler.transform(arr.reshape(-1, 1)).reshape(arr.shape)

    interpreter.set_tensor(input_index, arr.astype(np.float32))
    interpreter.invoke()
    saida = interpreter.get_tensor(output_index)[0][0]

    match = re.search(r'_(\d+\.\d+)\.npy$', nome_arq)
    if match:
        rotulo = float(match.group(1))
        nomes.append(nome_arq)
        y_real.append(rotulo)
        y_pred.append(saida)

# === METRICS ===
y_real = np.array(y_real)
y_pred = np.array(y_pred)
residuals = y_real - y_pred
absolute_error = np.abs(residuals)
relative_error = absolute_error / y_real * 100

mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
r2 = r2_score(y_real, y_pred)
mean_percent_error = np.mean(relative_error)

# === SAVE RESULTS ===
df = pd.DataFrame({
    "filename": nomes,
    "real_kWh": y_real,
    "predicted_kWh": y_pred,
    "absolute_error": absolute_error,
    "relative_error_%": relative_error
})
df.to_csv(os.path.join(PASTA_SAIDA, "results.csv"), index=False)
df.sort_values("real_kWh").to_csv(os.path.join(PASTA_SAIDA, "results_sorted.csv"), index=False)
df.sort_values("absolute_error", ascending=False).head(10).to_csv(os.path.join(PASTA_SAIDA, "top10_errors.csv"), index=False)

with open(os.path.join(PASTA_SAIDA, "metrics_test.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f} kWh\nRMSE: {rmse:.2f} kWh\nR²: {r2:.4f}\nMean Percent Error: {mean_percent_error:.2f}%\n")

# === PLOTS ===
plt.figure(); plt.plot(y_real, label="Real"); plt.plot(y_pred, label="Predicted")
plt.legend(); plt.title("Real vs Predicted (Line Plot)"); plt.grid(); plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "real_vs_predicted_line.png"))

plt.figure(); plt.scatter(y_real, y_pred, s=10); plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], 'r--')
plt.xlabel("Real (kWh)"); plt.ylabel("Predicted (kWh)"); plt.title("Scatter Plot"); plt.grid(); plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "real_vs_predicted_scatter.png"))

plt.figure(); sns.histplot(absolute_error, bins=40, kde=True)
plt.title("Histogram of Absolute Errors"); plt.xlabel("Absolute Error (kWh)"); plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "error_histogram.png"))

plt.figure(); sns.boxplot(y=relative_error)
plt.title("Boxplot of Relative Errors (%)"); plt.ylabel("Relative Error (%)"); plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "relative_error_boxplot.png"))

plt.figure(); plt.scatter(y_pred, residuals, s=10); plt.axhline(0, color='r', linestyle='--')
plt.title("Residuals vs Predicted"); plt.xlabel("Predicted (kWh)"); plt.ylabel("Residual (kWh)"); plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "residuals_vs_predicted.png"))

plt.figure(); plt.scatter(y_real, absolute_error, s=10)
plt.title("Absolute Error vs Real Value"); plt.xlabel("Real (kWh)"); plt.ylabel("Absolute Error (kWh)")
plt.tight_layout(); plt.savefig(os.path.join(PASTA_SAIDA, "abs_error_vs_real.png"))

# ECDF
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

x, y = ecdf(relative_error); plt.figure(); plt.plot(x, y, marker='.')
plt.title("ECDF of Relative Errors (%)"); plt.xlabel("Relative Error (%)"); plt.ylabel("ECDF")
plt.tight_layout(); plt.savefig(os.path.join(PASTA_SAIDA, "ecdf_relative_error.png"))

x, y = ecdf(absolute_error); plt.figure(); plt.plot(x, y, marker='.')
plt.title("ECDF of Absolute Errors"); plt.xlabel("Absolute Error (kWh)"); plt.ylabel("ECDF")
plt.tight_layout(); plt.savefig(os.path.join(PASTA_SAIDA, "error_cdf.png"))

plt.figure(); stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals"); plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "qq_plot_residuals.png"))

# MAE by decile
df["decile"] = pd.qcut(df["real_kWh"], 10, labels=False)
mae_decile = df.groupby("decile")["absolute_error"].mean().reset_index()
mae_decile.to_csv(os.path.join(PASTA_SAIDA, "mae_by_decile.csv"), index=False)

plt.figure(); plt.bar(mae_decile["decile"], mae_decile["absolute_error"])
plt.title("MAE by Decile of Real Consumption"); plt.xlabel("Decile"); plt.ylabel("MAE (kWh)")
plt.tight_layout(); plt.savefig(os.path.join(PASTA_SAIDA, "mae_by_decile.png"))

print("✅ Quantized model inference completed.")
