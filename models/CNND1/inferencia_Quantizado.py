import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === CONFIGURAÇÕES ===
PASTA_NPY = r"C:\projeto_artigo\mono\CNND1\com_interharmonicas\teste_npy"
MODELO_TFLITE = "modelo_final_quantizado/modelo_final.tflite"
PASTA_SAIDA = "resultado_quantizado_final"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# === CARREGAR MODELO ===
print(f"📦 Carregando modelo TFLite: {MODELO_TFLITE}")
interpreter = tf.lite.Interpreter(model_path=MODELO_TFLITE)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# === CARREGAR ARQUIVOS .NPY ===
arquivos_npy = sorted([f for f in os.listdir(PASTA_NPY) if f.endswith(".npy")])
print(f"🔍 {len(arquivos_npy)} arquivos encontrados.")

y_real = []
y_pred = []
nomes = []

for nome_arq in tqdm(arquivos_npy, desc="🔎 Inferindo"):
    caminho = os.path.join(PASTA_NPY, nome_arq)
    arr = np.load(caminho)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1, 1)
    elif arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], 1)
    
    interpreter.set_tensor(input_index, arr.astype(np.float32))
    interpreter.invoke()
    saida = interpreter.get_tensor(output_index)[0][0]
    rotulo = float(nome_arq.split("_")[-1].replace(".npy", ""))
    
    nomes.append(nome_arq)
    y_pred.append(saida)
    y_real.append(rotulo)

y_real = np.array(y_real)
y_pred = np.array(y_pred)
residuos = y_real - y_pred
erro_absoluto = np.abs(residuos)
erro_relativo = np.abs((y_real - y_pred) / y_real) * 100

# === SALVAR RESULTADOS ===
df = pd.DataFrame({
    "arquivo": nomes,
    "real_kWh": y_real,
    "previsto_kWh": y_pred,
    "erro_absoluto": erro_absoluto,
    "erro_relativo_%": erro_relativo
})
df.to_csv(os.path.join(PASTA_SAIDA, "resultados.csv"), index=False)

# Corrigido ordenado
df_corrigido = df.sort_values(by="real_kWh").reset_index(drop=True)
df_corrigido.to_csv(os.path.join(PASTA_SAIDA, "resultados_corrigidos.csv"), index=False)

# Top 10 erros absolutos
top10 = df.sort_values(by="erro_absoluto", ascending=False).head(10)
top10.to_csv(os.path.join(PASTA_SAIDA, "top10_errors.csv"), index=False)

# === MÉTRICAS ===
mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
r2 = r2_score(y_real, y_pred)
erro_percentual = np.mean(erro_relativo)

with open(os.path.join(PASTA_SAIDA, "metrics_test.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f} kWh\n")
    f.write(f"RMSE: {rmse:.2f} kWh\n")
    f.write(f"R²: {r2:.4f}\n")
    f.write(f"Erro Percentual Médio: {erro_percentual:.2f}%\n")

# === GRÁFICOS ===
plt.figure()
plt.plot(y_real, label="Real")
plt.plot(y_pred, label="Previsto")
plt.legend()
plt.title("Real vs Previsto (linha)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "real_vs_predicted_line.png"))

plt.figure()
plt.scatter(y_real, y_pred, s=10)
plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
plt.xlabel("Real (kWh)")
plt.ylabel("Previsto (kWh)")
plt.title("Real vs Previsto (scatter)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "real_vs_predicted_scatter.png"))

plt.figure()
sns.histplot(erro_absoluto, bins=40, kde=True)
plt.title("Histogram of Absolute Errors")
plt.xlabel("Absolute Error (kWh)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "error_histogram.png"))

plt.figure()
sns.boxplot(y=erro_relativo)
plt.title("Boxplot of Relative Errors (%)")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "relative_error_boxplot.png"))

plt.figure()
plt.scatter(y_pred, residuos, s=10)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Value")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "residuals_vs_predicted.png"))

plt.figure()
plt.scatter(y_real, erro_absoluto, s=10)
plt.xlabel("Real Value")
plt.ylabel("Absolute Error")
plt.title("Absolute Error vs Real Value")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "abs_error_vs_real.png"))

# ECDFs
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

x1, y1 = ecdf(erro_relativo)
plt.figure()
plt.plot(x1, y1, marker='.')
plt.title("ECDF of Relative Errors (%)")
plt.xlabel("Relative Error (%)")
plt.ylabel("ECDF")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "ecdf_relative_error.png"))

x2, y2 = ecdf(erro_absoluto)
plt.figure()
plt.plot(x2, y2, marker='.')
plt.title("CDF of Absolute Errors")
plt.xlabel("Absolute Error (kWh)")
plt.ylabel("CDF")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "error_cdf.png"))

# QQ-plot de resíduos
import scipy.stats as stats
import matplotlib.pyplot as plt

plt.figure()
stats.probplot(residuos, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "qq_plot_residuals.png"))

# MAE por decil
df_corrigido["decil"] = pd.qcut(df_corrigido["real_kWh"], 10, labels=False)
mae_por_decil = df_corrigido.groupby("decil")["erro_absoluto"].mean().reset_index()
mae_por_decil.to_csv(os.path.join(PASTA_SAIDA, "mae_by_decile.csv"), index=False)

plt.figure()
plt.bar(mae_por_decil["decil"], mae_por_decil["erro_absoluto"])
plt.xlabel("Decil of Real Value")
plt.ylabel("MAE (kWh)")
plt.title("MAE by Decile of Real Consumption")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "mae_by_decile.png"))

print("✅ Inferência e geração de resultados finalizada.")
