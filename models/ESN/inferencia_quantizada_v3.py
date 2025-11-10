# inferencia_segura_v3_quantizada_parallel.py
import os, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import resample
from scipy.stats import probplot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from reservoirpy.nodes import Reservoir
from joblib import Parallel, delayed
import tensorflow as tf

# ================== CONFIG ==================
CSV_TEST_DIR = r"C:\projeto_artigo\mono\TCN\com_interharmonicas\teste"
MODEL_DIR    = r"C:\projeto_artigo\mono\2_ESN\com_interharmonicas\modelo_treinado_seguro_v3"
TFLITE_PATH  = r"C:\projeto_artigo\mono\2_ESN\com_interharmonicas\modelo_treinado_seguro_v3_quantizado\ridge_model_esn_int8.tflite"
OUT_DIR      = r"C:\projeto_artigo\mono\2_ESN\com_interharmonicas\inferencia_segura_v3_quantizada"
TARGET_LEN   = 142560
UNITS        = 1000
SR           = 1.0
NORMALIZE    = True
N_JOBS       = 8

os.makedirs(OUT_DIR, exist_ok=True)

# ================== Helpers ==================
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

FP_TEST = {"units":UNITS, "sr":SR, "inputs":len(REF_COLS), "hash": fingerprint_res(UNITS, SR, len(REF_COLS))}
if FP_TEST != FP_TRAIN:
    raise RuntimeError(f"Reservatório diferente entre treino e inferência.\nTreino: {FP_TRAIN}\nInferência: {FP_TEST}")

DROP_TOKENS = ['energy','accum','kwh']

def padronizar_cols(df):
    cols_drop = [c for c in df.columns if any(t in c.lower() for t in DROP_TOKENS)]
    if cols_drop: df = df.drop(columns=cols_drop)
    df = df[[c for c in REF_COLS if c in df.columns]]
    if len(df.columns) != len(REF_COLS):
        raise ValueError("Colunas do CSV não batem com as do treino.")
    return df

def quantize_fp32_to_int8(x, scale, zp):
    q = np.round(x / scale + zp).astype(np.int8)
    return np.clip(q, -128, 127)

def dequantize_int8_to_fp32(q, scale, zp):
    return scale * (q.astype(np.int32) - zp)

def carregar_estado_esn(csv_path):
    df = pd.read_csv(csv_path)
    df = padronizar_cols(df)
    x = df.values.astype(np.float32)
    x = resample(x, TARGET_LEN).astype(np.float32)
    if NORMALIZE:
        x = (x - x.mean())/(x.std()+1e-8)
    res = Reservoir(units=UNITS, sr=SR, input_scaling=1.0, bias_scaling=0.0, seed=42)
    res.reset()
    return res.run(x)[-1].astype(np.float32)  # (UNITS,)

def inferir_um_arquivo(nome_csv):
    if not nome_csv.endswith(".csv"):
        return None
    try:
        caminho = os.path.join(CSV_TEST_DIR, nome_csv)
        state = carregar_estado_esn(caminho)             # (UNITS,)
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        in_scale, in_zp   = inp["quantization"]
        out_scale, out_zp = out["quantization"]

        xin = state[np.newaxis, :]                       # (1, UNITS) float32
        xq  = quantize_fp32_to_int8(xin, in_scale, in_zp)
        interpreter.set_tensor(inp["index"], xq)
        interpreter.invoke()
        yq = interpreter.get_tensor(out["index"])        # int8 (1,1)
        y  = float(dequantize_int8_to_fp32(yq, out_scale, out_zp)[0,0])
        y  = max(0.0, y)  # coerência física mínima (kWh >= 0) – a versão “corrigida” usa esse valor

        label = float(os.path.splitext(nome_csv)[0].split('_')[-1])
        return label, y
    except Exception as e:
        print(f"❌ Erro em {nome_csv}: {e}")
        return None

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name)); plt.close(fig)

# ================== Run ==================
if __name__ == "__main__":
    arquivos = sorted([f for f in os.listdir(CSV_TEST_DIR) if f.endswith(".csv")])
    if not arquivos:
        raise RuntimeError("Sem CSV no diretório de teste.")

    resultados = Parallel(n_jobs=N_JOBS, backend="loky", prefer="processes")(
        delayed(inferir_um_arquivo)(f) for f in tqdm(arquivos, desc="Inferência INT8 paralela (ESN + TFLite)")
    )
    resultados = [r for r in resultados if r is not None]
    if not resultados:
        raise RuntimeError("Nenhuma amostra processada.")

    y_true, y_pred_corr = map(np.array, zip(*resultados))  # y_pred já clipado ≥0 (corrigido)
    # Para também salvar a visão “crua”, vamos refazer sem o clip na métrica principal:
    # (Você pode pular isso e usar só o corrigido se preferir)
    # Aqui assumimos que o y_pred_corr é >=0; para “cru”, usamos a mesma série (ou remova o max(0) no worker se quiser estritamente cru).
    y_pred_cru = y_pred_corr.copy()

    # ===== DataFrames =====
    df = pd.DataFrame({"Real": y_true, "Predicted": y_pred_cru})
    df["Absolute_Error"]   = np.abs(df["Real"] - df["Predicted"])
    df["Residuals"]        = df["Real"] - df["Predicted"]
    df["Relative_Error(%)"]= (df["Absolute_Error"] / (df["Real"] + 1e-8)) * 100
    df.to_csv(os.path.join(OUT_DIR, "resultados.csv"), index=False)

    df_corr = pd.DataFrame({"Real": y_true, "Predicted": y_pred_corr})
    df_corr["Absolute_Error"]    = np.abs(df_corr["Real"] - df_corr["Predicted"])
    df_corr["Residuals"]         = df_corr["Real"] - df_corr["Predicted"]
    df_corr["Relative_Error(%)"] = (df_corr["Absolute_Error"] / (df_corr["Real"] + 1e-8)) * 100
    df_corr.to_csv(os.path.join(OUT_DIR, "resultados_corrigidos.csv"), index=False)

    # ===== Métricas (usando predição “crua”; troque para df_corr se preferir pós-clip) =====
    mae  = mean_absolute_error(df["Real"], df["Predicted"])
    rmse = np.sqrt(mean_squared_error(df["Real"], df["Predicted"]))
    r2   = r2_score(df["Real"], df["Predicted"])
    with open(os.path.join(OUT_DIR, "metrics_test.txt"), "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\n")

    # ===== Top 10 erros =====
    top10 = df.sort_values(by="Absolute_Error", ascending=False).head(10)
    top10.to_csv(os.path.join(OUT_DIR, "top10_errors.csv"), index=False)

    # ===== MAE por decil =====
    df['Decile'] = pd.qcut(df['Real'], 10, labels=False)
    mae_decile = df.groupby('Decile')["Absolute_Error"].mean()
    mae_decile.to_csv(os.path.join(OUT_DIR, "mae_by_decile.csv"))

    # ===== Gráficos (10) =====
    # 1) Absolute Error vs Real
    fig = plt.figure()
    plt.scatter(df["Real"], df["Absolute_Error"], alpha=0.6)
    plt.xlabel("Real Value (kWh)"); plt.ylabel("Absolute Error (kWh)")
    plt.title("Absolute Error vs Real Value"); plt.grid(True)
    save(fig, "abs_error_vs_real.png")

    # 2) ECDF of Relative Error
    fig = plt.figure()
    sorted_err = np.sort(df["Relative_Error(%)"])
    ecdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
    plt.plot(sorted_err, ecdf)
    plt.xlabel("Relative Error (%)"); plt.ylabel("ECDF")
    plt.title("ECDF of Relative Error (%)"); plt.grid(True)
    save(fig, "ecdf_relative_error.png")

    # 3) CDF of Absolute Error
    fig = plt.figure()
    sorted_abs = np.sort(df["Absolute_Error"])
    ecdf_abs = np.arange(1, len(sorted_abs)+1) / len(sorted_abs)
    plt.plot(sorted_abs, ecdf_abs)
    plt.xlabel("Absolute Error (kWh)"); plt.ylabel("CDF")
    plt.title("CDF of Absolute Error"); plt.grid(True)
    save(fig, "error_cdf.png")

    # 4) Histogram of Absolute Error
    fig = plt.figure()
    plt.hist(df["Absolute_Error"], bins=20)
    plt.xlabel("Absolute Error (kWh)"); plt.ylabel("Frequency")
    plt.title("Histogram of Absolute Error"); plt.grid(True)
    save(fig, "error_histogram.png")

    # 5) MAE by Decile
    fig = plt.figure()
    mae_decile.plot(kind='bar')
    plt.ylabel("MAE (kWh)"); plt.title("MAE by Decile of Real Value"); plt.grid(True)
    save(fig, "mae_by_decile.png")

    # 6) Q-Q Plot of Residuals
    fig = plt.figure()
    probplot(df["Residuals"], dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    save(fig, "qq_plot_residuals.png")

    # 7) Real vs Predicted Scatter
    fig = plt.figure()
    plt.scatter(df["Real"], df["Predicted"], alpha=0.6)
    mn, mx = df["Real"].min(), df["Real"].max()
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel("Real Value (kWh)"); plt.ylabel("Predicted (kWh)")
    plt.title("Real vs Predicted Scatter"); plt.grid(True)
    save(fig, "real_vs_predicted_scatter.png")

    # 8) Real vs Predicted Line
    fig = plt.figure(figsize=(10,4))
    plt.plot(df["Real"].reset_index(drop=True), label="Real")
    plt.plot(df["Predicted"].reset_index(drop=True), label="Predicted", alpha=0.7)
    plt.title("Real vs Predicted Line"); plt.xlabel("Sample Index"); plt.ylabel("kWh")
    plt.legend(); plt.grid(True)
    save(fig, "real_vs_predicted_line.png")

    # 9) Relative Error Boxplot
    fig = plt.figure()
    plt.boxplot(df["Relative_Error(%)"])
    plt.ylabel("Relative Error (%)"); plt.title("Boxplot of Relative Error (%)"); plt.grid(True)
    save(fig, "relative_error_boxplot.png")

    # 10) Residuals vs Predicted
    fig = plt.figure()
    plt.scatter(df["Predicted"], df["Residuals"], alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted (kWh)"); plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted"); plt.grid(True)
    save(fig, "residuals_vs_predicted.png")

    print(f"\n✅ Inferência INT8 concluída. R²={r2:.4f}")
    print("Arquivos gerados em:", OUT_DIR)
