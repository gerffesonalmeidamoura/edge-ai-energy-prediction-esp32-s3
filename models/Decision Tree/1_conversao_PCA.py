import os
import numpy as np
import pandas as pd
import glob
import pickle
import gc
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import time

PASTA_AMOSTRAS = r"C:\projeto_artigo\mono\TCN\com_interharmonicas\amostras_3dias_sem_harmonicas"
SAIDA = r"C:\projeto_artigo\mono\2_DECISION\2_com_interharmonicas\saida_pca_decision"
N_COMPONENTES = 50
BATCH_SIZE = 64
DOWNSAMPLE = 1
VARIANCIA_MINIMA_MONOTONICA = 1e-5

os.makedirs(SAIDA, exist_ok=True)

# Global
COLUNAS_COMUNS = None

def limpar_dataframe(df):
    df = df.iloc[:, 1:-1]  # Remove primeira e última coluna
    df = df.iloc[::DOWNSAMPLE, :]  # Downsampling
    df = df.loc[:, df.nunique() > 1]  # Remove constantes
    colunas_remover = [
        col for col in df.columns
        if (df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing)
        and df[col].std() < VARIANCIA_MINIMA_MONOTONICA
    ]
    return df.drop(columns=colunas_remover, errors='ignore')

def obter_colunas_comuns(arquivos):
    colunas_comuns = None
    for path in tqdm(arquivos, desc="📋 Buscando colunas comuns"):
        try:
            df = pd.read_csv(path, skiprows=1)
            df = limpar_dataframe(df)
            if colunas_comuns is None:
                colunas_comuns = set(df.columns)
            else:
                colunas_comuns &= set(df.columns)
        except Exception as e:
            print(f"⚠️ Erro lendo {os.path.basename(path)}: {e}")
    return sorted(list(colunas_comuns)) if colunas_comuns else []

def carregar_arquivo(path):
    try:
        df = pd.read_csv(path, skiprows=1)
        df = limpar_dataframe(df)
        df = df[COLUNAS_COMUNS]  # Usa só as colunas comuns
        if df.isnull().values.any() or np.isinf(df.values).any():
            return None, None
        vetor = df.to_numpy(dtype=np.float32).flatten()
        label = float(os.path.basename(path).split("_")[-1].replace(".csv", ""))
        return vetor, label
    except Exception as e:
        print(f"⚠️ Erro ao carregar {os.path.basename(path)}: {e}")
        return None, None

def ajustar_pca_incremental(arquivos):
    pca = IncrementalPCA(n_components=N_COMPONENTES)
    for i in tqdm(range(0, len(arquivos), BATCH_SIZE), desc="🔧 Ajuste PCA"):
        lote = []
        for path in arquivos[i:i+BATCH_SIZE]:
            vetor, _ = carregar_arquivo(path)
            if vetor is not None:
                lote.append(vetor)
        if lote:
            try:
                X = np.array(lote, dtype=np.float32)
                pca.partial_fit(X)
                del X
                del lote
                gc.collect()
            except Exception as e:
                print(f"⚠️ Erro no lote {i//BATCH_SIZE + 1}: {e}")
    return pca

def transformar_amostras(arquivos, pca):
    X_pca = []
    y = []
    for path in tqdm(arquivos, desc="🚀 Transformando com PCA"):
        vetor, label = carregar_arquivo(path)
        if vetor is not None:
            X_pca.append(pca.transform([vetor])[0])
            y.append(label)
    return np.array(X_pca, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    global COLUNAS_COMUNS

    print("🔍 Buscando arquivos...")
    arquivos = sorted(glob.glob(os.path.join(PASTA_AMOSTRAS, "amostra_*.csv")))
    print(f"🔍 {len(arquivos)} arquivos encontrados.")

    print("\n📋 Etapa 0: Identificando colunas comuns após limpeza...")
    COLUNAS_COMUNS = obter_colunas_comuns(arquivos)
    print(f"✅ {len(COLUNAS_COMUNS)} colunas comuns encontradas.")

    # Salvar colunas comuns para inferência futura
    with open(os.path.join(SAIDA, "colunas_comuns.txt"), "w") as f:
        for col in COLUNAS_COMUNS:
            f.write(col + "\n")
    print(f"📝 colunas_comuns.txt salvo em: {os.path.join(SAIDA, 'colunas_comuns.txt')}")

    t0 = time.perf_counter()
    print("\n⚙️ Etapa 1: Ajustando PCA incrementalmente com limpeza...")
    pca = ajustar_pca_incremental(arquivos)
    print(f"⏱️ Tempo Etapa 1: {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    print("\n⚙️ Etapa 2: Transformando os dados com PCA...")
    X_pca, y = transformar_amostras(arquivos, pca)
    print(f"⏱️ Tempo Etapa 2: {time.perf_counter() - t1:.2f}s")

    # Salvar resultados
    with open(os.path.join(SAIDA, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    np.save(os.path.join(SAIDA, "X_pca.npy"), X_pca)
    np.save(os.path.join(SAIDA, "y.npy"), y)

    print(f"\n✅ Conversão finalizada!")
    print(f"📦 PCA salvo em: {os.path.join(SAIDA, 'pca.pkl')}")
    print(f"📈 X salvo em: {os.path.join(SAIDA, 'X_pca.npy')} com shape {X_pca.shape}")
    print(f"🎯 y salvo em: {os.path.join(SAIDA, 'y.npy')} com shape {y.shape}")

if __name__ == "__main__":
    main()
