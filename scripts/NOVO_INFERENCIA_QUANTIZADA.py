import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import tensorflow as tf
import logging
from tqdm import tqdm

# ——— Logging ——————————————————————————————————————————————————————————————
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ——— Configurações ——————————————————————————————————————————————————————
MODEL_DIR    = "modelo_final_quantizado"
TEST_DIR     = "teste"
OUT_DIR      = "resultados_teste_quantizado"
SAMPLE_STEP  = 10  # downsampling usado no treinamento

os.makedirs(OUT_DIR, exist_ok=True)

# ——— Carregar TFLite e PCA ——————————————————————————————————————————————————
logging.info("📦 Carregando modelo TFLite e PCA")
interpreter = tf.lite.Interpreter(model_path=os.path.join(MODEL_DIR, "modelo_final_quantizado.tflite"))
interpreter.allocate_tensors()
inp_det  = interpreter.get_input_details()[0]
out_det  = interpreter.get_output_details()[0]
in_scale, in_zp   = inp_det['quantization']
out_scale, out_zp = out_det['quantization']

with open(os.path.join(MODEL_DIR, "pca.pkl"), "rb") as f:
    pca = pickle.load(f)
logging.info(f"🔧 PCA espera {pca.n_features_in_} features")

# ——— Inferência ————————————————————————————————————————————————————————
csv_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.csv")))
logging.info(f"🔍 Encontradas {len(csv_paths)} amostras em '{TEST_DIR}'")

results = []
for path in tqdm(csv_paths, desc="Inferindo (TFLite)"):
    name = os.path.basename(path)
    try:
        df = pd.read_csv(path)
        df = df.drop(df.columns[0], axis=1)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df.fillna(0.0)
        df = df.iloc[::SAMPLE_STEP, :].reset_index(drop=True)
        arr = df.to_numpy(dtype="float32").flatten().reshape(1, -1)
        if arr.shape[1] != pca.n_features_in_:
            logging.warning(f"Shape mismatch em {name}: {arr.shape[1]} != {pca.n_features_in_}")
            continue
        Xp = pca.transform(arr).astype("float32")
        Xi8 = np.clip(
            np.round(Xp / in_scale + in_zp),
            np.iinfo(inp_det['dtype']).min,
            np.iinfo(inp_det['dtype']).max
        ).astype(inp_det['dtype'])
        interpreter.set_tensor(inp_det['index'], Xi8)
        interpreter.invoke()
        raw = interpreter.get_tensor(out_det['index']).astype(np.int32)
        y_pred = float((raw - out_zp) * out_scale)
        y_true = float(name.split("_")[-1].replace(".csv",""))
        results.append((name, y_true, y_pred))
    except Exception as e:
        logging.error(f"Erro em {name}: {e}")

# ——— Criar DataFrame —————————————————————————————————————————————————————
df = pd.DataFrame(results, columns=["amostra","valor_real","valor_predito"])
df.to_csv(os.path.join(OUT_DIR, "resultados.csv"), index=False)

# ——— Correção polinomial grau 2 —————————————————————————————————————————
if len(df) > 1:
    coef2, coef1, coef0 = np.polyfit(df["valor_predito"], df["valor_real"], deg=2)
else:
    coef2, coef1, coef0 = 0.0, 1.0, 0.0
df["valor_predito_corrigido"] = (
    coef2*df["valor_predito"]**2 +
    coef1*df["valor_predito"] +
    coef0
)
df.to_csv(os.path.join(OUT_DIR, "resultados_corrigidos.csv"), index=False)
logging.info(f"📈 Correção: y={coef2:.6f}x²+{coef1:.6f}x+{coef0:.6f}")

# ——— Métricas ————————————————————————————————————————————————————————
df["abs_error"] = np.abs(df["valor_real"] - df["valor_predito_corrigido"])
residuals = df["valor_real"] - df["valor_predito_corrigido"]

mae  = df["abs_error"].mean() if len(df)>0 else np.nan
rmse = np.sqrt((df["abs_error"]**2).mean()) if len(df)>0 else np.nan
ss_res = (residuals**2).sum()
ss_tot = ((df["valor_real"]-df["valor_real"].mean())**2).sum()
r2 = 1 - ss_res/ss_tot if ss_tot!=0 else np.nan

with open(os.path.join(OUT_DIR, "metrics_test.txt"), "w") as f:
    f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\n")
logging.info(f"Metrics → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# ——— Geração de gráficos e arquivos adicionais ——————————————————————————
# 1) Scatter Real vs Predito
logging.info("Gerando real_vs_predicted_scatter.png")
plt.figure(figsize=(8,6))
plt.scatter(df["valor_real"], df["valor_predito_corrigido"], alpha=0.7)
mn, mx = df["valor_real"].min(), df["valor_real"].max()
plt.plot([mn, mx], [mn, mx], 'r--')
plt.xlabel("Real (kWh)"); plt.ylabel("Predito Corrigido (kWh)")
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "real_vs_predicted_scatter.png"))
plt.close()

# 2) Linha Real vs Predito
logging.info("Gerando real_vs_predicted_line.png")
plt.figure(figsize=(12,6))
plt.plot(df.index, df["valor_real"], label="Real", linewidth=2)
plt.plot(df.index, df["valor_predito_corrigido"], label="Predito", linewidth=2)
plt.xlabel("Amostra"); plt.ylabel("kWh")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "real_vs_predicted_line.png"))
plt.close()

# 3) ECDF dos Erros Absolutos
logging.info("Gerando error_cdf.png")
se = np.sort(df["abs_error"].values)
ecdf = np.arange(1, len(se)+1) / len(se)
plt.figure(figsize=(8,6))
plt.plot(se, ecdf, marker='.', linestyle='-')
plt.xlabel("Erro Absoluto (kWh)"); plt.ylabel("Prob. Cumulativa")
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "error_cdf.png"))
plt.close()

# 4) Histograma de Erros Absolutos
logging.info("Gerando error_histogram.png")
plt.figure(figsize=(8,6))
plt.hist(df["abs_error"], bins=30, edgecolor='black')
plt.xlabel("Erro Absoluto (kWh)"); plt.ylabel("Frequência")
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "error_histogram.png"))
plt.close()

# 5) Erro vs Valor Real
logging.info("Gerando abs_error_vs_real.png")
plt.figure(figsize=(8,6))
plt.scatter(df["valor_real"], df["abs_error"], alpha=0.7)
plt.xlabel("Real (kWh)"); plt.ylabel("Erro Absoluto")
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "abs_error_vs_real.png"))
plt.close()

# 6) Boxplot de Erros Relativos (%)
logging.info("Gerando relative_error_boxplot.png")
rel_err = df["abs_error"] / df["valor_real"] * 100
plt.figure(figsize=(8,6))
plt.boxplot(rel_err, patch_artist=True)
plt.ylabel("Erro Relativo (%)"); plt.title("Boxplot Erros Relativos")
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "relative_error_boxplot.png"))
plt.close()

# 7) Q‑Q Plot dos Resíduos
logging.info("Gerando qq_plot_residuals.png")
plt.figure(figsize=(8,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q‑Q Plot Resíduos"); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "qq_plot_residuals.png"))
plt.close()

# 8) ECDF de Erros Relativos
logging.info("Gerando ecdf_relative_error.png")
sr = np.sort(rel_err.values)
cr = np.arange(1, len(sr)+1) / len(sr)
plt.figure(figsize=(8,6))
plt.plot(sr, cr, marker='.', linestyle='-')
plt.xlabel("Erro Relativo (%)"); plt.ylabel("Prob. Cumulativa")
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ecdf_relative_error.png"))
plt.close()

# 9) Resíduos vs Predito
logging.info("Gerando residuals_vs_predicted.png")
plt.figure(figsize=(8,6))
plt.scatter(df["valor_predito_corrigido"], residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predito Corrigido"); plt.ylabel("Resíduo")
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residuals_vs_predicted.png"))
plt.close()

# 10) MAE por Decil
logging.info("Gerando mae_by_decile.png / mae_by_decile.csv")
df["decile"] = pd.qcut(df["valor_real"], 10, labels=False)
mae_dec = df.groupby("decile")["abs_error"].mean()
mae_dec.to_csv(os.path.join(OUT_DIR, "mae_by_decile.csv"), header=["MAE_kWh"])
plt.figure(figsize=(8,6))
mae_dec.plot(kind="bar")
plt.xlabel("Decil Real"); plt.ylabel("MAE (kWh)")
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mae_by_decile.png"))
plt.close()

# 11) Top‑10 Maiores Erros
logging.info("Gerando top10_errors.csv")
top10 = df.nlargest(10, "abs_error")[["amostra","valor_real","valor_predito_corrigido","abs_error"]]
top10.to_csv(os.path.join(OUT_DIR, "top10_errors.csv"), index=False)

logging.info(f"\n✅ Inferência e extração completas em '{OUT_DIR}'")
