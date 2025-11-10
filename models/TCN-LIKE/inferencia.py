import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

# Paths
pasta_modelo = r"C:\projeto_artigo\mono\2_TCN\com_interharmonicas\resultado_treinamento_TCN_filtros64"
pasta_teste = r"C:\projeto_artigo\mono\2_TCN\com_interharmonicas\convertidos_down_142560_TESTE"
os.makedirs(pasta_modelo, exist_ok=True)

# Load test samples
arquivos = sorted([f for f in os.listdir(pasta_teste) if f.endswith(".npy")])
print(f"🔍 {len(arquivos)} files found for inference.")

X_amostras, y_real = [], []
for nome in tqdm(arquivos, desc="🔄 Loading samples"):
    caminho = os.path.join(pasta_teste, nome)
    dado = np.load(caminho).reshape((142560, 1))
    X_amostras.append(dado)
    rotulo = float(nome.split("_")[-1].replace(".npy", ""))
    y_real.append(rotulo)

X_amostras = np.array(X_amostras)
y_real = np.array(y_real)
print(f"✅ Data loaded: X {X_amostras.shape}, y {y_real.shape}")

# Normalization (same as training)
scaler = StandardScaler()
X_amostras = scaler.fit_transform(X_amostras.reshape(-1, 1)).reshape(X_amostras.shape)

# Load model
modelo = tf.keras.models.load_model(os.path.join(pasta_modelo, "modelo_tcn.keras"))
print("📦 Model loaded successfully.")

# Inference
print("🚀 Running inference...")
y_pred = modelo.predict(X_amostras, batch_size=2).flatten()

# Evaluation
mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
r2 = r2_score(y_real, y_pred)
erro_percentual = np.mean(np.abs((y_real - y_pred) / y_real)) * 100

# Save metrics
with open(os.path.join(pasta_modelo, "metrics_test.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f} kWh\n")
    f.write(f"RMSE: {rmse:.2f} kWh\n")
    f.write(f"R²: {r2:.4f}\n")
    f.write(f"Mean Percent Error: {erro_percentual:.2f}%\n")

# Detailed results
df_resultados = pd.DataFrame({
    "real": y_real,
    "predicted": y_pred,
    "residual": y_real - y_pred,
    "absolute_error": np.abs(y_real - y_pred),
    "relative_error": np.abs((y_real - y_pred) / y_real) * 100
})
df_resultados.to_csv(os.path.join(pasta_modelo, "results.csv"), index=False)
df_resultados.sort_values(by="absolute_error", ascending=False).head(10).to_csv(os.path.join(pasta_modelo, "top10_errors.csv"), index=False)

# 📊 Plots
plt.figure(); plt.plot(y_real, label='Real'); plt.plot(y_pred, label='Predicted')
plt.title("Real vs Predicted (Line Plot)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(pasta_modelo, "real_vs_predicted_line.png"))

plt.figure(); plt.scatter(y_real, y_pred)
plt.xlabel("Real (kWh)"); plt.ylabel("Predicted (kWh)"); plt.title("Real vs Predicted (Scatter)")
plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], 'r--')
plt.tight_layout(); plt.savefig(os.path.join(pasta_modelo, "real_vs_predicted_scatter.png"))

plt.figure(); plt.hist(df_resultados['absolute_error'], bins=50)
plt.title("Histogram of Absolute Errors"); plt.xlabel("Absolute Error (kWh)")
plt.tight_layout(); plt.savefig(os.path.join(pasta_modelo, "error_histogram.png"))

plt.figure(); sns.boxplot(x=df_resultados['relative_error'])
plt.title("Boxplot of Relative Errors (%)"); plt.xlabel("Relative Error (%)")
plt.tight_layout(); plt.savefig(os.path.join(pasta_modelo, "relative_error_boxplot.png"))

ecdf_abs = ECDF(df_resultados['absolute_error'])
ecdf_rel = ECDF(df_resultados['relative_error'])

plt.figure(); plt.plot(ecdf_abs.x, ecdf_abs.y)
plt.title("ECDF of Absolute Errors"); plt.xlabel("Absolute Error (kWh)"); plt.ylabel("ECDF")
plt.tight_layout(); plt.savefig(os.path.join(pasta_modelo, "error_cdf.png"))

plt.figure(); plt.plot(ecdf_rel.x, ecdf_rel.y)
plt.title("ECDF of Relative Errors (%)"); plt.xlabel("Relative Error (%)"); plt.ylabel("ECDF")
plt.tight_layout(); plt.savefig(os.path.join(pasta_modelo, "ecdf_relative_error.png"))

plt.figure(); plt.scatter(y_real, np.abs(y_real - y_pred))
plt.xlabel("Real (kWh)"); plt.ylabel("Absolute Error (kWh)"); plt.title("Absolute Error vs Real Value")
plt.tight_layout(); plt.savefig(os.path.join(pasta_modelo, "abs_error_vs_real.png"))

plt.figure(); plt.scatter(y_pred, y_real - y_pred)
plt.xlabel("Predicted (kWh)"); plt.ylabel("Residual (kWh)"); plt.title("Residuals vs Predicted")
plt.axhline(0, color='red', linestyle='--'); plt.tight_layout()
plt.savefig(os.path.join(pasta_modelo, "residuals_vs_predicted.png"))

plt.figure(); stats.probplot(df_resultados['residual'], dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals"); plt.tight_layout()
plt.savefig(os.path.join(pasta_modelo, "qq_plot_residuals.png"))

# MAE by decile
df_resultados["decile"] = pd.qcut(df_resultados["real"], 10, labels=False)
df_decil = df_resultados.groupby("decile")["absolute_error"].mean().reset_index()
df_decil.to_csv(os.path.join(pasta_modelo, "mae_by_decile.csv"), index=False)

plt.figure(); plt.bar(df_decil["decile"], df_decil["absolute_error"])
plt.title("MAE by Decile of Real Consumption"); plt.xlabel("Decile"); plt.ylabel("MAE (kWh)")
plt.tight_layout(); plt.savefig(os.path.join(pasta_modelo, "mae_by_decile.png"))

print("✅ Final inference and result generation completed.")
