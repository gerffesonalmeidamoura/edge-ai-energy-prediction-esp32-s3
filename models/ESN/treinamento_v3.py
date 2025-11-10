import os, json, hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from reservoirpy.nodes import Reservoir
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
import multiprocessing

# ===== CONFIG =====
CSV_DIR   = r"C:\projeto_artigo\mono\TCN\com_interharmonicas\amostras_3dias_sem_harmonicas"
OUT_DIR   = r"C:\projeto_artigo\mono\2_ESN\com_interharmonicas\modelo_treinado_seguro_v3"
TARGET_LEN = 142560
UNITS = 1000
SR = 1.0
NORMALIZE = True
N_JOBS = 14
TOKENS_DROP = ['energy','accum','kwh']  # lowercase

os.makedirs(OUT_DIR, exist_ok=True)

def limpar_e_padronizar(df, ref_cols=None):
    cols_drop = [c for c in df.columns if any(t in c.lower() for t in TOKENS_DROP)]
    if cols_drop:
        df = df.drop(columns=cols_drop)
    if ref_cols is None:
        ref_cols = list(df.columns)
    # Reordena e filtra exatamente para as colunas de referência
    df = df[[c for c in ref_cols if c in df.columns]]
    # Se faltar alguma coluna, aborta explicitamente
    if len(df.columns) != len(ref_cols):
        raise ValueError(f"Colunas faltando. Esperado {len(ref_cols)}, obtido {len(df.columns)}.")
    return df, ref_cols

def processar_amostra(nome_arquivo, ref_cols):
    if not nome_arquivo.endswith(".csv"):
        return None
    try:
        caminho = os.path.join(CSV_DIR, nome_arquivo)
        df = pd.read_csv(caminho)
        df, _ = limpar_e_padronizar(df, ref_cols)

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
        print(f"❌ Erro em {nome_arquivo}: {e}")
        return None

def fingerprint_res(units, sr, d_inputs):
    # Fingerprint independente de acesso a W/Win:
    # roda o mesmo reservatório em um input sintético determinístico
    rng = np.random.default_rng(123)
    U = rng.standard_normal((512, d_inputs)).astype(np.float32)
    U = (U - U.mean())/(U.std()+1e-8)
    res = Reservoir(units=units, sr=sr, input_scaling=1.0, bias_scaling=0.0, seed=42)
    res.reset()
    s = res.run(U)[-1].astype(np.float32)  # (units,)
    h = hashlib.sha256(s.tobytes()).hexdigest()
    return {"units": units, "sr": sr, "inputs": d_inputs, "hash": h}

if __name__ == "__main__":
    if os.name == "nt":
        multiprocessing.freeze_support()

    arquivos = sorted([f for f in os.listdir(CSV_DIR) if f.endswith(".csv")])
    if not arquivos:
        raise RuntimeError("Sem CSV no diretório de treino.")

    # 1) Descobre e fixa o esquema de colunas
    df0 = pd.read_csv(os.path.join(CSV_DIR, arquivos[0]))
    df0, ref_cols = limpar_e_padronizar(df0, None)
    with open(os.path.join(OUT_DIR, "columns.json"), "w", encoding="utf-8") as f:
        json.dump(ref_cols, f, ensure_ascii=False, indent=2)
    print(f"✔ Esquema de colunas salvo ({len(ref_cols)} cols).")

    # 2) Fingerprint do reservatório (depende de D!)
    fp = fingerprint_res(UNITS, SR, len(ref_cols))
    with open(os.path.join(OUT_DIR, "reservoir_fingerprint.json"), "w") as f:
        json.dump(fp, f, indent=2)
    print("✔ Fingerprint salvo:", fp)

    # 3) Extrai estados
    resultados = Parallel(n_jobs=N_JOBS)(
        delayed(processar_amostra)(nome, ref_cols) for nome in tqdm(arquivos, desc="Processando treino")
    )
    resultados = [r for r in resultados if r is not None]
    states, labels = zip(*resultados)
    X = np.vstack(states)
    y = np.array(labels, dtype=np.float32)

    # 4) Treino com GridSearch simples
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    grid = GridSearchCV(Ridge(), {"alpha":[0.01,0.1,1.0,10.0,100.0]}, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    alpha = grid.best_params_["alpha"]
    print("✔ Melhor alpha:", alpha)

    modelo = Ridge(alpha=alpha).fit(X_train, y_train)
    y_train_pred = modelo.predict(X_train); y_val_pred = modelo.predict(X_val)

    # 5) Modelo final
    modelo_final = Ridge(alpha=alpha).fit(X, y)
    joblib.dump(modelo_final, os.path.join(OUT_DIR, "ridge_model.pkl"))
    with open(os.path.join(OUT_DIR, "alpha.txt"), "w") as f: f.write(str(alpha))

    # 6) Métricas
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
        f.write(f"alpha: {alpha}\n")

    print("✅ Treino seguro concluído e artefatos salvos em:", OUT_DIR)
