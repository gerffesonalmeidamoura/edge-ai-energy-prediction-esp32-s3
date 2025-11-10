import os
import pandas as pd
import glob
from itertools import combinations
from tqdm import tqdm

# Diretórios
pasta_entrada = "dados_originais"
pasta_saida  = "amostras_3dias_sem_harmonicas"
os.makedirs(pasta_saida, exist_ok=True)

dias_por_amostra = 3

# Lista todos os CSV
arquivos_csv = sorted(glob.glob(os.path.join(pasta_entrada, "*.csv")))
print(f"🔍 Encontrados {len(arquivos_csv)} arquivos em '{pasta_entrada}'")

# Todas as combinações de 3 arquivos
combos = list(combinations(arquivos_csv, dias_por_amostra))
print(f"🧮 Total de combinações possíveis: {len(combos)}")

for idx, trio in tqdm(enumerate(combos), total=len(combos)):
    dfs = []
    offset_l1 = 0.0
    offset_all = 0.0

    for caminho in trio:
        # 1) Lê CSV (header na 1ª linha)
        df = pd.read_csv(caminho)
        # 2) Remove a coluna de tempo/hora (primeira coluna)
        df = df.drop(df.columns[0], axis=1)

        # 3) Acumula Energy(kWh) ALL
        if "Energy(kWh) ALL" in df.columns:
            # ajusta para continuar de onde parou
            df["Energy(kWh) ALL"] = df["Energy(kWh) ALL"] + offset_all
            # atualiza offset para o próximo dia
            offset_all = df["Energy(kWh) ALL"].iloc[-1]

        # 4) Acumula Energy(kWh) L1
        if "Energy(kWh) L1" in df.columns:
            df["Energy(kWh) L1"] = df["Energy(kWh) L1"] + offset_l1
            offset_l1 = df["Energy(kWh) L1"].iloc[-1]

        dfs.append(df)

    # 5) Concatena os 3 dias
    amostra = pd.concat(dfs, ignore_index=True)

    # 6) Nome de saída com valor final de ALL
    nome_saida = f"amostra_{idx:04d}_{offset_all:.2f}.csv"
    caminho_saida = os.path.join(pasta_saida, nome_saida)
    amostra.to_csv(caminho_saida, index=False)

print(f"📁 Geradas {len(combos)} amostras de 3 dias em '{pasta_saida}'")
