#!/usr/bin/env python3
# inferencia_cnnd1_full.py

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─── IMPORTS PARA OS GRÁFICOS E RELATÓRIOS ───────────────────────────────────────
import pandas as pd
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

# ─── CONFIGURAÇÃO ───────────────────────────────────────────────────────────────
MODEL_DIR   = "resultado_treinamento_142560"
MODEL_PATH  = os.path.join(MODEL_DIR, "modelo_final.keras")
DATA_DIR    = "teste_npy"     # pasta dos .npy com shape (142560,)
OUTPUT_DIR  = MODEL_DIR
TIMESTEPS   = 142560

# ─── 1) Carrega o modelo ────────────────────────────────────────────────────────
print(f"\n📦 Carregando modelo: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# ─── 2) Lista e carrega os .npy de teste ─────────────────────────────────────────
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".npy"))
print(f"🔍 Encontrados {len(files)} arquivos em '{DATA_DIR}'\n")

X_list = []
y_list = []
for fname in tqdm(files, desc="Carregando NPY"):
    fp = os.path.join(DATA_DIR, fname)
    arr = np.load(fp)
    # checa shape
    if arr.ndim != 1 or arr.shape[0] != TIMESTEPS:
        raise ValueError(f"{fname}: esperado (142560,), mas veio {arr.shape}")
    X_list.append(arr.reshape(TIMESTEPS, 1))
    # extrai valor real do nome "..._<valor>.npy"
    val = float(fname.rsplit("_", 1)[1].replace(".npy", ""))
    y_list.append(val)

X_test = np.stack(X_list, axis=0).astype(np.float32)  # (n_amostras, 142560, 1)
y_test = np.array(y_list, dtype=np.float32)
print(f"\n✅ X_test: {X_test.shape}   y_test: {y_test.shape}")

# ─── 3) Inferência ──────────────────────────────────────────────────────────────
print("\n🚀 Inferindo...")
y_pred = model.predict(X_test, batch_size=32).flatten()

# ─── 4) Métricas básicas ───────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

metrics_txt = os.path.join(OUTPUT_DIR, "inference_metrics.txt")
with open(metrics_txt, "w") as f:
    f.write(f"MAE:  {mae:.4f} kWh\n")
    f.write(f"RMSE: {rmse:.4f} kWh\n")
    f.write(f"R2:   {r2:.4f}\n")
print(f"\n✅ Métricas salvas em: {metrics_txt}")
print(f"   MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

# ─── 5) Scatter real vs predito ────────────────────────────────────────────────
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
mn, mx = y_test.min(), y_test.max()
plt.plot([mn,mx],[mn,mx], "r--")
plt.xlabel("Real (kWh)")
plt.ylabel("Predito (kWh)")
plt.title("Real vs Predito")
plt.grid(True)
plt.tight_layout()
scatter_fp = os.path.join(OUTPUT_DIR, "scatter_real_predito.png")
plt.savefig(scatter_fp)
plt.close()
print(f"✅ Scatter salvo em: {scatter_fp}")

# ─── 6) Monta DataFrame de resultados ───────────────────────────────────────────
df = pd.DataFrame({
    "real":           y_test,
    "previsto":       y_pred,
})
df["erro"]            = df.real - df.previsto
df["erro_absoluto"]   = df.erro.abs()
df["erro_relativo_%"] = df.erro_absoluto / df.real * 100

# salva CSV completo e top10 erros
res_full = os.path.join(OUTPUT_DIR, "resultados.csv")
df.to_csv(res_full, index=False)
top10 = df.nlargest(10, "erro_absoluto")
top10_fp = os.path.join(OUTPUT_DIR, "top10_errors.csv")
top10.to_csv(top10_fp, index=False)
print(f"✅ CSVs de resultados salvos em: {res_full}  e  {top10_fp}")

# ─── 7) Gráfico série real x previsto ──────────────────────────────────────────
line_fp = os.path.join(OUTPUT_DIR, "real_vs_predicted_line.png")
plt.figure(figsize=(8,4))
plt.plot(df.real.values, label="Real")
plt.plot(df.previsto.values, label="Previsto")
plt.title("Real vs Previsto")
plt.legend()
plt.tight_layout()
plt.savefig(line_fp)
plt.close()
print(f"✅ Gráfico time-series salvo em: {line_fp}")

# ─── 8) Histograma de erro absoluto ────────────────────────────────────────────
hist_err_fp = os.path.join(OUTPUT_DIR, "error_histogram.png")
plt.figure(figsize=(6,4))
plt.hist(df.erro_absoluto, bins=50)
plt.title("Histograma de Erro Absoluto")
plt.tight_layout()
plt.savefig(hist_err_fp)
plt.close()
print(f"✅ Histograma de erro salvo em: {hist_err_fp}")

# ─── 9) Boxplot de erro relativo ────────────────────────────────────────────────
box_fp = os.path.join(OUTPUT_DIR, "relative_error_boxplot.png")
plt.figure(figsize=(6,4))
sns.boxplot(x=df["erro_relativo_%"])
plt.title("Boxplot Erro Relativo (%)")
plt.tight_layout()
plt.savefig(box_fp)
plt.close()
print(f"✅ Boxplot de erro relativo salvo em: {box_fp}")

# ─── 10) ECDFs ────────────────────────────────────────────────────────────────
ecdf_abs = ECDF(df.erro_absoluto)
ecdf_rel = ECDF(df["erro_relativo_%"])
ecdf_abs_fp = os.path.join(OUTPUT_DIR, "ecdf_error_absolute.png")
ecdf_rel_fp = os.path.join(OUTPUT_DIR, "ecdf_error_relative.png")

plt.figure(figsize=(6,4))
plt.plot(ecdf_abs.x, ecdf_abs.y)
plt.title("ECDF Erro Absoluto")
plt.tight_layout()
plt.savefig(ecdf_abs_fp)
plt.close()
plt.figure(figsize=(6,4))
plt.plot(ecdf_rel.x, ecdf_rel.y)
plt.title("ECDF Erro Relativo (%)")
plt.tight_layout()
plt.savefig(ecdf_rel_fp)
plt.close()
print(f"✅ ECDFs salvos em: {ecdf_abs_fp}, {ecdf_rel_fp}")

# ─── 11) Erro vs real e resíduos ────────────────────────────────────────────────
abs_vs_real_fp = os.path.join(OUTPUT_DIR, "abs_error_vs_real.png")
residuals_fp    = os.path.join(OUTPUT_DIR, "residuals_vs_predicted.png")

plt.figure(figsize=(6,4))
plt.scatter(df.real, df.erro_absoluto, alpha=0.3)
plt.xlabel("Real")
plt.ylabel("Erro Absoluto")
plt.title("Erro Absoluto vs Real")
plt.tight_layout()
plt.savefig(abs_vs_real_fp)
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(df.previsto, df.erro, alpha=0.3)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Previsto")
plt.ylabel("Resíduo")
plt.title("Resíduos vs Previsto")
plt.tight_layout()
plt.savefig(residuals_fp)
plt.close()
print(f"✅ Gráficos de resíduos salvos em: {abs_vs_real_fp}, {residuals_fp}")

# ─── 12) QQ‐Plot ────────────────────────────────────────────────────────────────
qq_fp = os.path.join(OUTPUT_DIR, "qq_plot_residuals.png")
plt.figure(figsize=(6,6))
stats.probplot(df.erro, dist="norm", plot=plt)
plt.title("QQ‑Plot dos Resíduos")
plt.tight_layout()
plt.savefig(qq_fp)
plt.close()
print(f"✅ QQ‑Plot salvo em: {qq_fp}")

# ─── 13) MAE por decil ──────────────────────────────────────────────────────────
df["decil"] = pd.qcut(df.real, 10, labels=False)
df_decil = df.groupby("decil")["erro_absoluto"].mean().reset_index()
deciles_fp = os.path.join(OUTPUT_DIR, "mae_by_decile.csv")
df_decil.to_csv(deciles_fp, index=False)

# gráfico de barras
mae_deciles_fp = os.path.join(OUTPUT_DIR, "mae_by_decile.png")
plt.figure(figsize=(6,4))
plt.bar(df_decil.decil, df_decil.erro_absoluto)
plt.xlabel("Decil")
plt.ylabel("MAE")
plt.title("MAE por Decil de Consumo Real")
plt.tight_layout()
plt.savefig(mae_deciles_fp)
plt.close()
print(f"✅ MAE por decil salvo em: {deciles_fp}, {mae_deciles_fp}")

print("\n🎉 Inferência + relatório completo! Confira os 15 arquivos em:", OUTPUT_DIR)
