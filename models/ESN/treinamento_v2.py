import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from reservoirpy.nodes import Reservoir
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
import multiprocessing

# === CONFIGURAÇÕES ===
CSV_DIR   = r"C:\projeto_artigo\mono\TCN\com_interharmonicas\amostras_3dias_sem_harmonicas"
OUT_DIR   = r"C:\projeto_artigo\mono\2_ESN\com_interharmonicas\modelo_treinado_seguro"
TARGET_LEN = 142560
UNITS = 1000
NORMALIZE = True
SR = 1.0
N_JOBS = 16

os.makedirs(OUT_DIR, exist_ok=True)

def processar_amostra(nome_arquivo):
    if not nome_arquivo.endswith(".csv"):
        return None
    try:
        caminho = os.path.join(CSV_DIR, nome_arquivo)
        df = pd.read_csv(caminho)

        colunas_remover = [col for col in df.columns if any(x in col.lower() for x in ['energy', 'accum', 'kwh'])]
        df = df.drop(columns=colunas_remover)

        dados = df.values.astype(np.float32)
        dados_rs = resample(dados, TARGET_LEN).astype(np.float32)

        if NORMALIZE:
            dados_rs = (dados_rs - dados_rs.mean()) / (dados_rs.std() + 1e-8)

        res = Reservoir(units=UNITS, sr=SR, input_scaling=1.0, bias_scaling=0.0, seed=42)
        res.reset()
        estado_final = res.run(dados_rs)[-1].astype(np.float32)

        label = float(os.path.splitext(nome_arquivo)[0].split('_')[-1])
        return estado_final, label

    except Exception as e:
        print(f"❌ Erro no arquivo {nome_arquivo}: {e}")
        return None

if __name__ == "__main__":
    if os.name == "nt":
        multiprocessing.freeze_support()

    print(f"🔧 Lendo e processando arquivos CSV em: {CSV_DIR}")
    arquivos = sorted(os.listdir(CSV_DIR))

    resultados = Parallel(n_jobs=N_JOBS)(
        delayed(processar_amostra)(nome) for nome in tqdm(arquivos, desc="Processando CSVs")
    )
    resultados = [r for r in resultados if r is not None]

    states, labels = zip(*resultados)
    X = np.vstack(states)
    y = np.array(labels, dtype=np.float32)

    print(f"🔧 Dividindo dados e treinando modelo Ridge…")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = Ridge(alpha=1.0)
    modelo.fit(X_train, y_train)

    y_train_pred = modelo.predict(X_train)
    y_val_pred   = modelo.predict(X_val)

    modelo_final = Ridge(alpha=1.0)
    modelo_final.fit(X, y)
    joblib.dump(modelo_final, os.path.join(OUT_DIR, "ridge_model.pkl"))

    mae_train  = mean_absolute_error(y_train, y_train_pred)
    mae_val    = mean_absolute_error(y_val, y_val_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_val   = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_train   = r2_score(y_train, y_train_pred)
    r2_val     = r2_score(y_val, y_val_pred)

    with open(os.path.join(OUT_DIR, "metrics_train.txt"), "w") as f:
        f.write(f"[Train] MAE: {mae_train:.4f}\n")
        f.write(f"[Train] RMSE: {rmse_train:.4f}\n")
        f.write(f"[Train] R²: {r2_train:.4f}\n\n")
        f.write(f"[Validation] MAE: {mae_val:.4f}\n")
        f.write(f"[Validation] RMSE: {rmse_val:.4f}\n")
        f.write(f"[Validation] R²: {r2_val:.4f}\n")

    print(f"\n✅ Treinamento concluído!")
    print(f"   MAE (train)  = {mae_train:.2f}")
    print(f"   MAE (val)    = {mae_val:.2f}")
    print(f"   RMSE (train) = {rmse_train:.2f}")
    print(f"   RMSE (val)   = {rmse_val:.2f}")
    print(f"   Modelo salvo em: {OUT_DIR}")

    # GRÁFICOS
    plt.figure()
    plt.bar(["Train", "Validation"], [rmse_train, rmse_val], color=["blue", "orange"])
    plt.ylabel("RMSE (kWh)")
    plt.title("Training vs Validation Loss (RMSE)")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "training_validation_loss.png"))
    plt.close()

    plt.figure()
    plt.bar(["Train", "Validation"], [mae_train, mae_val], color=["green", "red"])
    plt.ylabel("MAE (kWh)")
    plt.title("Training vs Validation MAE")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "training_validation_mae.png"))
    plt.close()
