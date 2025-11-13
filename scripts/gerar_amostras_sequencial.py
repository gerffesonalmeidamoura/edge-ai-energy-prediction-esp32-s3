import numpy as np
import os
import glob
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import random
import traceback

# Configurações
pasta_csv_diarios = "dados_originais"
pasta_saida = "dados_teste_3_dias"
os.makedirs(pasta_saida, exist_ok=True)

# Parâmetro: QUANTAS AMOSTRAS GERAR?
N_amostras = 286

# Listar CSVs
arquivos_csv = sorted(glob.glob(f"{pasta_csv_diarios}/*.csv"))
print(f"\n🔍 Encontrados {len(arquivos_csv)} arquivos de 1 dia")

# Nome da coluna de energia
nome_coluna_energia = "Energy(kWh) ALL"

# --- Cache local ---
_local_cache = {}

def carregar_csv(path_csv):
    if path_csv in _local_cache:
        return _local_cache[path_csv]

    df_check = pd.read_csv(path_csv)

    if df_check.shape[0] != 1440:
        raise ValueError(f"⚠️ Arquivo {os.path.basename(path_csv)} com {df_check.shape[0]} linhas (esperado: 1440)")

    if 'Unnamed: 0' in df_check.columns:
        df_check = df_check.drop(columns=['Unnamed: 0'])

    if nome_coluna_energia not in df_check.columns:
        raise ValueError(f"⚠️ Coluna de energia '{nome_coluna_energia}' não encontrada em {os.path.basename(path_csv)}")

    _local_cache[path_csv] = df_check
    return df_check

# Geração sequencial das combinações
combinacoes_lista = list(combinations(arquivos_csv, 3))
random.shuffle(combinacoes_lista)

total_combinacoes = len(combinacoes_lista)
print(f"🔢 Total de combinações possíveis: {total_combinacoes}")

if N_amostras > total_combinacoes:
    N_amostras = total_combinacoes
    print(f"⚠️ Reduzido para {N_amostras} (total de combinações)")

print(f"🎯 Gerando {N_amostras} amostras...\n")

nomes_arquivos_csv_gerados = []

for idx, combinacao in tqdm(list(enumerate(combinacoes_lista[:N_amostras]))):
    try:
        dfs = []
        label_total = 0.0
        erro_na_combinacao = False

        for path_csv in combinacao:
            try:
                df = carregar_csv(path_csv)

                df_temp = df.loc[:, ~df.columns.str.contains("TIME", case=False)]
                arr = df_temp.values.flatten().astype(np.float32)

                if arr.size == 0:
                    raise ValueError(f"Arquivo {os.path.basename(path_csv)} está vazio após remoção de TIME")

                dfs.append(arr)
                label_total += df[nome_coluna_energia].iloc[-1]

            except Exception as e:
                print(f"⚠️ Erro em {os.path.basename(path_csv)}: {e}")
                erro_na_combinacao = True
                break  # pula a combinação inteira

        if erro_na_combinacao:
            continue  # não tenta salvar nada

        arr_final = np.concatenate(dfs).astype(np.float32)

        if arr_final.size == 0:
            raise ValueError(f"Amostra {idx} resultou em array final vazio.")

        label_total_mes = label_total * 10
        nome_saida = f"amostra_{idx:04d}_{label_total_mes:.2f}.csv"
        path_saida = os.path.join(pasta_saida, nome_saida)

        tmp_path = path_saida + ".tmp"
        np.savetxt(tmp_path, arr_final[np.newaxis], delimiter=",", fmt="%.6f")
        os.replace(tmp_path, path_saida)

        nomes_arquivos_csv_gerados.append(nome_saida)

    except Exception as e:
        print(f"\n⚠️ Erro grave na combinação {idx}: {e}")
        traceback.print_exc()

print(f"\n✅ Geração concluída: {len(nomes_arquivos_csv_gerados)} amostras OK")
