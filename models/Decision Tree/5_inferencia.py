import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Caminho base
pasta_base = r"C:\projeto_artigo\mono\2_DECISION\2_com_interharmonicas\saida_pca_decision_teste"
os.makedirs(pasta_base, exist_ok=True)

# Carregar modelo
with open(os.path.join(r"C:\projeto_artigo\mono\2_DECISION\2_com_interharmonicas\saida_pca_decision", "model_decision.pkl"), "rb") as f:
    model = pickle.load(f)

# Carregar dados
X = np.load(os.path.join(pasta_base, "X_pca.npy"))
y = np.load(os.path.join(pasta_base, "y.npy"))

# Inferência
y_pred = model.predict(X)

# Erros
erro_abs = np.abs(y - y_pred)
erro_rel = np.abs((y - y_pred) / y) * 100

# Métricas
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Resultados
print(f"📊 MAE: {mae:.2f} kWh")
print(f"📊 RMSE: {rmse:.2f} kWh")
print(f"📊 R²: {r2:.4f}")

# Salvar métricas
with open(os.path.join(pasta_base, "metrics_test.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.4f}\n")

# Salvar CSV
df = pd.DataFrame({
    "real": y,
    "previsto": y_pred,
    "erro_abs": erro_abs,
    "erro_perc": erro_rel
})
df.to_csv(os.path.join(pasta_base, "resultados.csv"), index=False)

# Top 10 erros
df.sort_values(by="erro_abs", ascending=False).head(10).to_csv(os.path.join(pasta_base, "top10_erros.csv"), index=False)

# Gráficos
plt.figure()
plt.scatter(y, erro_abs, alpha=0.5)
plt.xlabel("Valor Real (kWh)")
plt.ylabel("Erro Absoluto (kWh)")
plt.title("Absolute Error vs Real Value")
plt.grid(True)
plt.savefig(os.path.join(pasta_base, "abs_error_vs_real.png"))
plt.close()

plt.figure()
plt.hist(erro_abs, bins=50)
plt.title("Histogram of Absolute Errors")
plt.xlabel("Erro Absoluto (kWh)")
plt.ylabel("Frequência")
plt.grid(True)
plt.savefig(os.path.join(pasta_base, "error_histogram.png"))
plt.close()

plt.figure()
plt.hist(erro_rel, bins=50)
plt.title("Histogram of Relative Errors (%)")
plt.xlabel("Erro Relativo (%)")
plt.ylabel("Frequência")
plt.grid(True)
plt.savefig(os.path.join(pasta_base, "error_cdf.png"))
plt.close()

plt.figure()
plt.plot(y, label="Real")
plt.plot(y_pred, label="Previsto")
plt.title("Real vs Predicted (Linha)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(pasta_base, "real_vs_predicted_line.png"))
plt.close()

plt.figure()
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Real (kWh)")
plt.ylabel("Previsto (kWh)")
plt.title("Real vs Predicted (Scatter)")
plt.grid(True)
plt.savefig(os.path.join(pasta_base, "real_vs_predicted_scatter.png"))
plt.close()

plt.figure()
plt.boxplot(erro_rel)
plt.title("Boxplot of Relative Errors (%)")
plt.savefig(os.path.join(pasta_base, "relative_error_boxplot.png"))
plt.close()

plt.figure()
plt.scatter(y_pred, y - y_pred, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Valor Previsto (kWh)")
plt.ylabel("Resíduo (Real - Previsto)")
plt.title("Residuals vs Predicted")
plt.grid(True)
plt.savefig(os.path.join(pasta_base, "residuals_vs_predicted.png"))
plt.close()

print("✅ Inferência finalizada e resultados salvos.")
