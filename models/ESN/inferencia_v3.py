import os, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import resample
from scipy.stats import probplot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from reservoirpy.nodes import Reservoir
from joblib import load, Parallel, delayed

# ===== CONFIG =====
CSV_TEST_DIR = r"C:\projeto_artigo\mono\TCN\com_interharmonicas\teste"
MODEL_DIR    = r"C:\projeto_artigo\mono\2_ESN\com_interharmonicas\modelo_treinado_seguro_v3"
OUT_DIR      = r"C:\projeto_artigo\mono\2_ESN\com_interharmonicas\inferencia_segura_v3"
TARGET_LEN   = 142560
UNITS = 1000
SR = 1.0
NORMALIZE = True
N_JOBS = 14

os.makedirs(OUT_DIR, exist_ok=True)

# ===== Utils =====
def fingerprint_res(units, sr, d_inputs):
    rng = np.random.default_rng(123)
    U = rng.standard_normal((512, d_inputs)).astype(np.float32)
    U = (U - U.mean())/(U.std()+1e-8)
    res = Reservoir(units=units, sr=sr, input_scaling=1.0, bias_scaling=0.0, seed=42)
    res.reset()
    s = res.run(U)[-1].astype(np.float32)
    return hashlib.sha256(s.tobytes()).hexdigest()

with open(os.path.join(MODEL_DIR, "columns.json"), "r", encoding="utf-8") as f:
    REF_COLS = json.load(f)
with open(os.path.join(MODEL_DIR, "reservoir_fingerprint.json"), "r") as f:
    FP_TRAIN = json.load(f)

# Checagem de fingerprint (unidades/sr/inputs)
FP_TEST = {"units":UNITS, "sr":SR, "inputs":len(REF_COLS), "hash": fingerprint_res(UNITS, SR, len(REF_COLS))}
if FP_TEST != FP_TRAIN:
    raise RuntimeError(f"Reservatório diferente entre treino e inferência.\nTreino: {FP_TRAIN}\nInferência: {FP_TEST}")

def padronizar_cols(df):
    drop_tokens = ['energy','accum','kwh']
    cols_drop = [c for c in df.columns if any(t in c.lower() for t in drop_tokens)]
    if cols_drop: df = df.drop(columns=cols_drop)
    df = df[[c for c in REF_COLS if c in df.columns]]
    if len(df.columns) != len(REF_COLS):
        raise ValueError("Colunas do CSV não batem com as do treino.")
    return df

def processar_um_csv(nome_arquivo):
    if not nome_arquivo.endswith(".csv"):
        return None
    try:
        caminho = os.path.join(CSV_TEST_DIR, nome_arquivo)
        df = pd.read_csv(caminho)
        df = padronizar_cols(df)

        x = df.values.astype(np.float32)
        x = resample(x, TARGET_LEN).astype(np.float32)
        if NORMALIZE:
            x = (x - x.mean())/(x.std()+1e-8)

        res = Reservoir(units=UNITS, sr=SR, input_scaling=1.0, bias_scaling=0.0, seed=42)
        res.reset()
        estado_final = res.run(x)[-1].astype(np.float32)

        label = float(os.path.splitext(nome_arquivo)[0].split('_')[-1])
        return estado_final, label
    except Exception as e:
        print(f"❌ {nome_arquivo}: {e}")
        return None

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name)); plt.close(fig)

# ===== INFERÊNCIA =====
if __name__ == "__main__":
    arquivos = sorted([f for f in os.listdir(CSV_TEST_DIR) if f.endswith(".csv")])
    if not arquivos:
        raise RuntimeError("Sem CSV no diretório de teste.")

    resultados = Parallel(n_jobs=N_JOBS)(
        delayed(processar_um_csv)(f) for f in tqdm(arquivos, desc="Inferindo estados")
    )
    resultados = [r for r in resultados if r is not None]
    if not resultados:
        raise RuntimeError("Nenhuma amostra processada.")

    X, y_true = map(np.array, zip(*resultados))
    modelo = load(os.path.join(MODEL_DIR, "ridge_model.pkl"))
    y_pred = modelo.predict(X)

    # CSV “cru”
    df = pd.DataFrame({"Real": y_true, "Predicted": y_pred})
    df["Absolute_Error"] = np.abs(df["Real"] - df["Predicted"])
    df["Residuals"] = df["Real"] - df["Predicted"]
    df["Relative_Error(%)"] = (df["Absolute_Error"] / (df["Real"] + 1e-8)) * 100
    df.to_csv(os.path.join(OUT_DIR, "resultados.csv"), index=False)

    # Versão “corrigida” (clip físico kWh>=0) + salva
    df_corr = df.copy()
    df_corr["Predicted"] = np.clip(df_corr["Predicted"], 0, None)
    df_corr["Absolute_Error"] = np.abs(df_corr["Real"] - df_corr["Predicted"])
    df_corr["Residuals"] = df_corr["Real"] - df_corr["Predicted"]
    df_corr["Relative_Error(%)"] = (df_corr["Absolute_Error"] / (df_corr["Real"] + 1e-8)) * 100
    df_corr.to_csv(os.path.join(OUT_DIR, "resultados_corrigidos.csv"), index=False)

    # Métricas (no conjunto “cru” – mantenha como preferir)
    mae  = mean_absolute_error(df["Real"], df["Predicted"])
    rmse = np.sqrt(mean_squared_error(df["Real"], df["Predicted"]))
    r2   = r2_score(df["Real"], df["Predicted"])

    with open(os.path.join(OUT_DIR, "metrics_test.txt"), "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\n")

    # Top 10 erros
    top10 = df.sort_values(by="Absolute_Error", ascending=False).head(10)
    top10.to_csv(os.path.join(OUT_DIR, "top10_errors.csv"), index=False)

    # MAE por decil
    df['Decile'] = pd.qcut(df['Real'], 10, labels=False)
    mae_decile = df.groupby('Decile')["Absolute_Error"].mean()
    mae_decile.to_csv(os.path.join(OUT_DIR, "mae_by_decile.csv"))

    # ===== GRÁFICOS (iguais aos do anexo) =====

    # 1) abs_error_vs_real.png
    fig = plt.figure()
    plt.scatter(df["Real"], df["Absolute_Error"], alpha=0.6)
    plt.xlabel("Real Value (kWh)")
    plt.ylabel("Absolute Error (kWh)")
    plt.title("Absolute Error vs Real Value")
    plt.grid(True)
    save(fig, "abs_error_vs_real.png")

    # 2) ecdf_relative_error.png
    fig = plt.figure()
    sorted_err = np.sort(df["Relative_Error(%)"])
    ecdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
    plt.plot(sorted_err, ecdf)
    plt.xlabel("Relative Error (%)")
    plt.ylabel("ECDF")
    plt.title("ECDF of Relative Error (%)")
    plt.grid(True)
    save(fig, "ecdf_relative_error.png")

    # 3) error_cdf.png (CDF do erro absoluto)
    fig = plt.figure()
    sorted_abs = np.sort(df["Absolute_Error"])
    ecdf_abs = np.arange(1, len(sorted_abs)+1) / len(sorted_abs)
    plt.plot(sorted_abs, ecdf_abs)
    plt.xlabel("Absolute Error (kWh)")
    plt.ylabel("CDF")
    plt.title("CDF of Absolute Error")
    plt.grid(True)
    save(fig, "error_cdf.png")

    # 4) error_histogram.png
    fig = plt.figure()
    plt.hist(df["Absolute_Error"], bins=20)
    plt.xlabel("Absolute Error (kWh)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Absolute Error")
    plt.grid(True)
    save(fig, "error_histogram.png")

    # 5) mae_by_decile.png
    fig = plt.figure()
    mae_decile.plot(kind='bar')
    plt.ylabel("MAE (kWh)")
    plt.title("MAE by Decile of Real Value")
    plt.grid(True)
    save(fig, "mae_by_decile.png")

    # 6) qq_plot_residuals.png
    fig = plt.figure()
    probplot(df["Residuals"], dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    save(fig, "qq_plot_residuals.png")

    # 7) real_vs_predicted_scatter.png
    fig = plt.figure()
    plt.scatter(df["Real"], df["Predicted"], alpha=0.6)
    mn, mx = df["Real"].min(), df["Real"].max()
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel("Real Value (kWh)")
    plt.ylabel("Predicted (kWh)")
    plt.title("Real vs Predicted Scatter")
    plt.grid(True)
    save(fig, "real_vs_predicted_scatter.png")

    # 8) real_vs_predicted_line.png
    fig = plt.figure(figsize=(10,4))
    plt.plot(df["Real"].reset_index(drop=True), label="Real")
    plt.plot(df["Predicted"].reset_index(drop=True), label="Predicted", alpha=0.7)
    plt.title("Real vs Predicted Line")
    plt.xlabel("Sample Index")
    plt.ylabel("kWh")
    plt.legend()
    plt.grid(True)
    save(fig, "real_vs_predicted_line.png")

    # 9) relative_error_boxplot.png
    fig = plt.figure()
    plt.boxplot(df["Relative_Error(%)"])
    plt.ylabel("Relative Error (%)")
    plt.title("Boxplot of Relative Error (%)")
    plt.grid(True)
    save(fig, "relative_error_boxplot.png")

    # 10) residuals_vs_predicted.png
    fig = plt.figure()
    plt.scatter(df["Predicted"], df["Residuals"], alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted (kWh)")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.grid(True)
    save(fig, "residuals_vs_predicted.png")

    print(f"\n✅ Inferência segura concluída. R²={r2:.4f}")
    print("Arquivos gerados em:", OUT_DIR)
