import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from cffi import FFI
import math
import scipy.stats as stats

# Caminhos
base_dir = r"C:\projeto_artigo\mono\2_DECISION\2_com_interharmonicas"
pasta_modelo = os.path.join(base_dir, "saida_pca_decision")
pasta_teste = os.path.join(base_dir, "saida_pca_decision_teste")
pasta_resultado = os.path.join(base_dir, "resultado_inferencia_quantizado")  # <-- ALTERADO
os.makedirs(pasta_resultado, exist_ok=True)

# Carregar dados
X = np.load(os.path.join(pasta_teste, "X_pca.npy")).astype("float32")
y_real = np.load(os.path.join(pasta_teste, "y.npy")).astype("float32")

# Compilar código C
with open(os.path.join(pasta_modelo, "modelo_decision_inferencia.c"), "r") as f:
    codigo_c = f.read()
ffi = FFI()
ffi.cdef("float predict(float f[]);")
C = ffi.verify(codigo_c, extra_compile_args=["-O3"])

# Inferência
y_pred = []
for i in range(X.shape[0]):
    entrada = ffi.new("float[]", X[i].tolist())
    y_pred.append(C.predict(entrada))
y_pred = np.array(y_pred, dtype="float32")

# Cálculo de erros
erro_abs = np.abs(y_real - y_pred)
erro_rel = np.abs((y_real - y_pred) / y_real) * 100

# Métricas
mae = mean_absolute_error(y_real, y_pred)
rmse = math.sqrt(mean_squared_error(y_real, y_pred))
r2 = r2_score(y_real, y_pred)
erro_percentual = np.mean(erro_rel)

# Salvar métricas
with open(os.path.join(pasta_resultado, "metrics_test.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R²: {r2:.4f}\n")
    f.write(f"Erro Percentual Médio: {erro_percentual:.2f}%\n")

# DataFrame resultados
df = pd.DataFrame({
    "real": y_real,
    "previsto": y_pred,
    "erro_abs": erro_abs,
    "erro_rel_percent": erro_rel
})
df.to_csv(os.path.join(pasta_resultado, "resultados_corrigidos.csv"), index=False)

# Top 10 erros
df.sort_values("erro_abs", ascending=False).head(10).to_csv(
    os.path.join(pasta_resultado, "top10_errors.csv"), index=False)

# Dispersão real vs previsto
plt.figure()
plt.scatter(y_real, y_pred, alpha=0.6)
plt.xlabel("Valor Real (kWh)")
plt.ylabel("Valor Previsto (kWh)")
plt.title("Real vs Previsto (Scatter)")
plt.grid(True)
plt.savefig(os.path.join(pasta_resultado, "real_vs_predicted_scatter.png"))
plt.close()

# Linha real vs previsto
plt.figure()
plt.plot(y_real, label="Real", alpha=0.7)
plt.plot(y_pred, label="Previsto", alpha=0.7)
plt.xlabel("Amostra")
plt.ylabel("Consumo (kWh)")
plt.legend()
plt.title("Real vs Previsto (Linha)")
plt.grid(True)
plt.savefig(os.path.join(pasta_resultado, "real_vs_predicted_line.png"))
plt.close()

# ECDF - Erro absoluto
sorted_errors = np.sort(erro_abs)
ecdf_y = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
plt.figure()
plt.plot(sorted_errors, ecdf_y, marker=".")
plt.xlabel("Erro Absoluto (kWh)")
plt.ylabel("Frequência Acumulada")
plt.title("ECDF of Absolute Errors")
plt.grid(True)
plt.savefig(os.path.join(pasta_resultado, "error_cdf.png"))
plt.close()

# ECDF - Erro relativo
sorted_rel_errors = np.sort(erro_rel)
ecdf_y_rel = np.arange(1, len(sorted_rel_errors)+1) / len(sorted_rel_errors)
plt.figure()
plt.plot(sorted_rel_errors, ecdf_y_rel, marker=".")
plt.xlabel("Erro Relativo (%)")
plt.ylabel("Frequência Acumulada")
plt.title("ECDF of Relative Errors")
plt.grid(True)
plt.savefig(os.path.join(pasta_resultado, "ecdf_relative_error.png"))
plt.close()

# Histograma dos erros absolutos
plt.figure()
plt.hist(erro_abs, bins=30, alpha=0.7)
plt.xlabel("Erro Absoluto (kWh)")
plt.ylabel("Frequência")
plt.title("Histograma dos Erros Absolutos")
plt.grid(True)
plt.savefig(os.path.join(pasta_resultado, "error_histogram.png"))
plt.close()

# Boxplot de erro relativo
plt.figure()
plt.boxplot(erro_rel, vert=True, patch_artist=True)
plt.title("Boxplot of Relative Errors (%)")
plt.ylabel("Erro Relativo (%)")
plt.grid(True)
plt.savefig(os.path.join(pasta_resultado, "relative_error_boxplot.png"))
plt.close()

# Erro absoluto vs real
plt.figure()
plt.scatter(y_real, erro_abs, alpha=0.6)
plt.xlabel("Valor Real (kWh)")
plt.ylabel("Erro Absoluto (kWh)")
plt.title("Erro Absoluto vs Valor Real")
plt.grid(True)
plt.savefig(os.path.join(pasta_resultado, "abs_error_vs_real.png"))
plt.close()

# Resíduos vs previsto
residuos = y_real - y_pred
plt.figure()
plt.scatter(y_pred, residuos, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Previsto (kWh)")
plt.ylabel("Resíduo (kWh)")
plt.title("Resíduos vs Previsto")
plt.grid(True)
plt.savefig(os.path.join(pasta_resultado, "residuals_vs_predicted.png"))
plt.close()

# QQ Plot dos resíduos
plt.figure()
stats.probplot(residuos, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.grid(True)
plt.savefig(os.path.join(pasta_resultado, "qq_plot_residuals.png"))
plt.close()

# MAE por decil
df["decil"] = pd.qcut(df["real"], 10, labels=False)
mae_por_decil = df.groupby("decil").apply(lambda d: mean_absolute_error(d["real"], d["previsto"]))
mae_por_decil.plot(kind="bar")
plt.xlabel("Decil de Valor Real")
plt.ylabel("MAE (kWh)")
plt.title("MAE by Decile of Real Consumption")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pasta_resultado, "mae_by_decile.png"))
plt.close()
mae_por_decil.to_csv(os.path.join(pasta_resultado, "mae_by_decile.csv"))

print("✅ Inferência com modelo C finalizada. Todos os arquivos foram gerados.")
