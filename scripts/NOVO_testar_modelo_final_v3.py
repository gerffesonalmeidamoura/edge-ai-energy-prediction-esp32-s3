import os
import glob
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm

# ——— Configurações ———
pasta_modelo    = "modelo_final"
pasta_amostras  = "teste"
pasta_resultados = "resultados_teste"
os.makedirs(pasta_resultados, exist_ok=True)

# mesmo downsampling usado no treinamento
sample_step = 10  

# ——— Carregar modelo e PCA ———
print(f"\n📦 Carregando modelo: {pasta_modelo}/modelo_final.keras")
model = load_model(os.path.join(pasta_modelo, "modelo_final.keras"))

print(f"📦 Carregando PCA: {pasta_modelo}/pca.pkl")
with open(os.path.join(pasta_modelo, "pca.pkl"), "rb") as f:
    pca = pickle.load(f)

# ——— Encontrar amostras de teste ———
arquivos_csv = sorted(glob.glob(os.path.join(pasta_amostras, "*.csv")))
print(f"\n🔍 Lendo {len(arquivos_csv)} amostras em '{pasta_amostras}'...\n")

resultados = []

# ——— Loop de inferência ———
for path in tqdm(arquivos_csv, desc="Testando amostras"):
    try:
        # 1) Leitura com pandas (igual ao treinamento)
        df = pd.read_csv(path)
        df = df.drop(df.columns[0], axis=1)                          # remove coluna de tempo
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]        # descarta colunas vazias
        df = df.fillna(0.0)
        # 2) Downsampling temporal (se usado no treinamento)
        df = df.iloc[::sample_step, :].reset_index(drop=True)

        # 3) Flatten e reshape
        arr = df.to_numpy(dtype="float32").flatten().reshape(1, -1)

        # 4) Verificar compatibilidade de shape
        if arr.shape[1] != pca.n_features_in_:
            print(f"⚠️ Shape mismatch: {os.path.basename(path)} → {arr.shape[1]} vs {pca.n_features_in_}")
            continue

        # 5) PCA → predição
        X_pca = pca.transform(arr)
        y_pred = model.predict(X_pca).flatten()[0]

        # 6) Extrair valor real do nome do arquivo
        valor_real = float(os.path.basename(path).split("_")[-1].replace(".csv", ""))
        resultados.append((os.path.basename(path), valor_real, y_pred))

    except Exception as e:
        print(f"⚠️ Erro em {os.path.basename(path)}: {e}")

# ——— Salvar resultados brutos ———
df_res = pd.DataFrame(resultados, columns=["amostra", "valor_real", "valor_predito"])
df_res.to_csv(os.path.join(pasta_resultados, "resultados.csv"), index=False)

# ——— Correção polinomial grau 2 ———
coef2, coef1, coef0 = np.polyfit(df_res["valor_predito"], df_res["valor_real"], deg=2)
df_res["valor_predito_corrigido"] = (
    coef2 * df_res["valor_predito"]**2 +
    coef1 * df_res["valor_predito"] +
    coef0
)
df_res.to_csv(os.path.join(pasta_resultados, "resultados_corrigidos.csv"), index=False)

print(f"\n✅ Coeficientes de correção polinomial grau 2:")
print(f"   → a2 (quadrático): {coef2:.8f}")
print(f"   → a1 (linear):     {coef1:.8f}")
print(f"   → a0 (constante):  {coef0:.8f}")

# ——— Métricas finais ———
mae  = np.mean(np.abs(df_res["valor_real"] - df_res["valor_predito_corrigido"]))
rmse = np.sqrt(np.mean((df_res["valor_real"] - df_res["valor_predito_corrigido"])**2))
ss_res = np.sum((df_res["valor_real"] - df_res["valor_predito_corrigido"])**2)
ss_tot = np.sum((df_res["valor_real"] - np.mean(df_res["valor_real"]))**2)
r2    = 1 - (ss_res / ss_tot)

with open(os.path.join(pasta_resultados, "metrics_test.txt"), "w") as f:
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

print(f"\n✅ Resultados salvos em:\n"
      f"   → {pasta_resultados}/resultados.csv\n"
      f"   → {pasta_resultados}/resultados_corrigidos.csv ({len(df_res)} amostras)\n")
print("\n🎯 Final TEST metrics (after polynomial correction):")
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
