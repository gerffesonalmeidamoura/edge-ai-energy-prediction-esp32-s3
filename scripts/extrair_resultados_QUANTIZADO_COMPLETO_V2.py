import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# === CONFIGURACAO PARA O MODELO QUANTIZADO ===
pasta_modelo = "modelo_final_quantizado"
pasta_resultados_teste = "resultados_teste_quantizado"
os.makedirs(pasta_modelo, exist_ok=True)
os.makedirs(pasta_resultados_teste, exist_ok=True)

# === CARREGAR RESULTADOS ===
df = pd.read_csv(f"{pasta_resultados_teste}/resultados_corrigidos.csv")

# === METRICAS ===
mae = np.mean(np.abs(df["valor_real"] - df["valor_predito_corrigido"]))
rmse = np.sqrt(np.mean((df["valor_real"] - df["valor_predito_corrigido"]) ** 2))
ss_res = np.sum((df["valor_real"] - df["valor_predito_corrigido"]) ** 2)
ss_tot = np.sum((df["valor_real"] - np.mean(df["valor_real"])) ** 2)
r2 = 1 - (ss_res / ss_tot)

with open(f"{pasta_resultados_teste}/metrics_test.txt", "w") as f:
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

# === ERROS ===
df["abs_error"] = np.abs(df["valor_real"] - df["valor_predito_corrigido"])
df["relative_error_%"] = df["abs_error"] / df["valor_real"] * 100
residuals = df["valor_real"] - df["valor_predito_corrigido"]

# 1) Scatter Real vs Predito
plt.figure(figsize=(8, 6))
plt.scatter(df["valor_real"], df["valor_predito_corrigido"], alpha=0.7)
plt.plot([df["valor_real"].min(), df["valor_real"].max()],
         [df["valor_real"].min(), df["valor_real"].max()],
         color="red", linestyle="--", label="Ideal (y = x)")
plt.xlabel("Real Value (kWh)")
plt.ylabel("Predicted Value (kWh)")
plt.title("Real vs Predicted (Test Set, Corrected)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_resultados_teste}/real_vs_predicted_scatter.png")
plt.close()

# 2) Linha Real vs Predito
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["valor_real"], label="Real Value", linewidth=2)
plt.plot(df.index, df["valor_predito_corrigido"], label="Predicted (Corrected)", linewidth=2)
plt.xlabel("Sample Index")
plt.ylabel("Consumption (kWh)")
plt.title("Real vs Predicted per Sample (Corrected)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_resultados_teste}/real_vs_predicted_line.png")
plt.close()

# 3) CDF dos Erros Absolutos
sorted_errors = np.sort(df["abs_error"])
cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
plt.figure(figsize=(8, 6))
plt.plot(sorted_errors, cdf, marker=".", linestyle="-")
plt.xlabel("Absolute Error (kWh)")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Absolute Errors (Test Set, Corrected)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_resultados_teste}/error_cdf.png")
plt.close()

# 4) Histograma dos Erros Absolutos
plt.figure(figsize=(8, 6))
plt.hist(df["abs_error"], bins=30, edgecolor='black')
plt.xlabel("Absolute Error (kWh)")
plt.ylabel("Frequency")
plt.title("Histogram of Absolute Errors (Test Set, Corrected)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_resultados_teste}/error_histogram.png")
plt.close()

# 5) Erro Absoluto vs Valor Real
plt.figure(figsize=(8, 6))
plt.scatter(df["valor_real"], df["abs_error"], alpha=0.7)
plt.xlabel("Real Value (kWh)")
plt.ylabel("Absolute Error (kWh)")
plt.title("Absolute Error vs Real Value (Test Set, Corrected)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_resultados_teste}/abs_error_vs_real.png")
plt.close()

# 6) Boxplot dos Erros Relativos
plt.figure(figsize=(8, 6))
plt.boxplot(df["relative_error_%"], vert=True, patch_artist=True)
plt.ylabel("Relative Error (%)")
plt.title("Boxplot of Relative Errors (%) (Test Set, Corrected)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_resultados_teste}/relative_error_boxplot.png")
plt.close()

# 7) Q-Q Plot dos Resíduos
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals (Test Set, Corrected)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_resultados_teste}/qq_plot_residuals.png")
plt.close()

# 8) ECDF dos Erros Relativos
sorted_rel_err = np.sort(df["relative_error_%"])
cdf_rel = np.arange(1, len(sorted_rel_err) + 1) / len(sorted_rel_err)
plt.figure(figsize=(8, 6))
plt.plot(sorted_rel_err, cdf_rel, marker=".", linestyle="-")
plt.xlabel("Relative Error (%)")
plt.ylabel("Cumulative Probability")
plt.title("ECDF of Relative Errors (%) (Test Set, Corrected)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_resultados_teste}/ecdf_relative_error.png")
plt.close()

# 9) Resíduos vs Predito
plt.figure(figsize=(8, 6))
plt.scatter(df["valor_predito_corrigido"], residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Value (kWh)")
plt.ylabel("Residual (Real - Predicted)")
plt.title("Residuals vs Predicted (Test Set, Corrected)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_resultados_teste}/residuals_vs_predicted.png")
plt.close()

# 10) MAE por Decil
df["decile"] = pd.qcut(df["valor_real"], 10, labels=False)
df_mae_deciles = df.groupby("decile").apply(lambda x: np.mean(np.abs(x["valor_real"] - x["valor_predito_corrigido"])))
df_mae_deciles.to_csv(f"{pasta_resultados_teste}/mae_by_decile.csv", header=["MAE_kWh"])

plt.figure(figsize=(8, 6))
df_mae_deciles.plot(kind="bar")
plt.xlabel("Decile of Real Value")
plt.ylabel("Mean Absolute Error (kWh)")
plt.title("MAE by Decile of Real Consumption (Test Set, Corrected)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pasta_resultados_teste}/mae_by_decile.png")
plt.close()

# 11) Top 10 maiores erros
top10 = df.copy()
top10 = top10.sort_values(by="abs_error", ascending=False).head(10)
top10[["amostra", "valor_real", "valor_predito_corrigido", "abs_error", "relative_error_%"]].to_csv(
    f"{pasta_resultados_teste}/top10_errors.csv", index=False
)

# Banner Final
print("\n✅ Test results from quantized model extracted successfully!")
print(f"📁 Outputs in '{pasta_modelo}/' and '{pasta_resultados_teste}/' (corrected)")
print("\n🎯 Final TEST metrics (after polynomial correction):")
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
