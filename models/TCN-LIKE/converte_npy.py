import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Diretórios
pasta_csv = r"C:\projeto_artigo\mono\CNND1\sem_harmonicas\dados_teste_3_dias"
pasta_saida = os.path.join(pasta_csv, "npy_convertidos")
os.makedirs(pasta_saida, exist_ok=True)

# Arquivos CSV
arquivos = sorted([f for f in os.listdir(pasta_csv) if f.endswith(".csv")])

print(f"🔍 Encontrados {len(arquivos)} arquivos para processar...")

for nome in tqdm(arquivos):
    try:
        caminho = os.path.join(pasta_csv, nome)
        df = pd.read_csv(caminho)

        # Verificar colunas obrigatórias
        if not {"timestamp", "EnergiaAcumulada_KWh"}.issubset(df.columns):
            print(f"⚠️ Colunas esperadas não encontradas em: {nome}")
            continue

        # Remover timestamp e rótulo
        df_filtrado = df.drop(columns=["timestamp", "EnergiaAcumulada_KWh"])

        # Flatten
        vetor_flat = df_filtrado.values.flatten()

        # Verificar se contém NaN ou Inf
        if np.isnan(vetor_flat).any() or np.isinf(vetor_flat).any():
            print(f"❌ Ignorado (NaN/Inf): {nome}")
            continue

        # Gerar nome e salvar
        nome_base = os.path.splitext(nome)[0]
        np.save(os.path.join(pasta_saida, f"{nome_base}.npy"), vetor_flat.astype(np.float32))

    except Exception as e:
        print(f"❌ Erro ao processar {nome}: {e}")

print("✅ Conversão finalizada. Arquivos .npy prontos para uso.")
