import os
import numpy as np
import pandas as pd
import glob
import pickle
from tqdm import tqdm

# === CONFIGURAÇÕES ===
PASTA_AMOSTRAS = r"C:\projeto_artigo\mono\TCN\com_interharmonicas\teste"
SAIDA = r"C:\projeto_artigo\mono\2_DECISION\2_com_interharmonicas\saida_pca_decision_teste"
PCA_TREINADO_PATH = r"C:\projeto_artigo\mono\2_DECISION\2_com_interharmonicas\saida_pca_decision\pca.pkl"
COLUNAS_COMUNS_PATH = os.path.join(os.path.dirname(PCA_TREINADO_PATH), "colunas_comuns.txt")
DOWNSAMPLE = 1
VARIANCIA_MINIMA_MONOTONICA = 1e-5

# Criar diretório de saída
os.makedirs(SAIDA, exist_ok=True)

# Carregar PCA treinado
with open(PCA_TREINADO_PATH, "rb") as f:
    pca = pickle.load(f)

# Carregar colunas comuns geradas no passo 1
with open(COLUNAS_COMUNS_PATH, "r") as f:
    COLUNAS_COMUNS = [linha.strip() for linha in f if linha.strip()]

print(f"📋 {len(COLUNAS_COMUNS)} colunas carregadas do passo 1.")

# Função de limpeza (igual ao treino)
def limpar_dataframe(df):
    df = df.iloc[:, 1:-1]  # Remove primeira e última coluna
    df = df.iloc[::DOWNSAMPLE, :]  # Downsampling
    df = df.loc[:, df.nunique() > 1]  # Remove colunas constantes
    colunas_remover = [
        col for col in df.columns
        if (df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing)
        and df[col].std() < VARIANCIA_MINIMA_MONOTONICA
    ]
    df = df.drop(columns=colunas_remover, errors='ignore')
    return df

# Coletar arquivos CSV
arquivos_csv = sorted(glob.glob(os.path.join(PASTA_AMOSTRAS, "amostra_*.csv")))
print(f"\n🔍 {len(arquivos_csv)} arquivos encontrados.")

# Aplicar transformação PCA
X_pca = []
y_full = []

for arq in tqdm(arquivos_csv, desc="🚀 Aplicando PCA (teste)"):
    try:
        df = pd.read_csv(arq, skiprows=1)
        df = limpar_dataframe(df)

        # Garantir que só as colunas comuns sejam usadas
        df = df[[col for col in COLUNAS_COMUNS if col in df.columns]]

        # Verificar dados inválidos
        if df.isnull().values.any() or np.isinf(df.values).any():
            raise ValueError("Dados inválidos (NaN ou Inf)")

        vetor = df.to_numpy(dtype=np.float32).flatten().reshape(1, -1)
        if vetor.shape[1] != pca.components_.shape[1]:
            raise ValueError(f"Shape incompatível com PCA ({vetor.shape[1]} vs {pca.components_.shape[1]})")

        X_pca.append(pca.transform(vetor)[0])

        # Rótulo
        label = float(os.path.basename(arq).split("_")[-1].replace(".csv", ""))
        y_full.append(label)

    except Exception as e:
        print(f"⚠️ Erro ao processar {os.path.basename(arq)}: {e}")

# Salvar resultados
X_pca = np.array(X_pca, dtype=np.float32)
y_full = np.array(y_full, dtype=np.float32)

np.save(os.path.join(SAIDA, "X_pca.npy"), X_pca)
np.save(os.path.join(SAIDA, "y.npy"), y_full)

print(f"\n✅ PCA aplicado com sucesso.")
print(f"📈 X_pca shape: {X_pca.shape}")
print(f"🎯 y shape: {y_full.shape}")
print(f"📁 Dados salvos em: {SAIDA}")
