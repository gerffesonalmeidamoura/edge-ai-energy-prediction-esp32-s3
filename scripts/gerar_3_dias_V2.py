import numpy as np
import os
import glob
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import multiprocessing as mp
from math import comb
import time
import random
import traceback

# Configurações
pasta_csv_diarios = "dados_originais"
pasta_saida = "dados_teste_3_dias"
os.makedirs(pasta_saida, exist_ok=True)

# Quantidade de amostras a gerar
N_amostras = 1330  # ajuste conforme desejado

# Nome da coluna com energia
nome_coluna_energia = "Energy(kWh) ALL"

# Cache local por processo
_local_cache = {}

def carregar_csv(path_csv):
    if path_csv in _local_cache:
        return _local_cache[path_csv]

    df = pd.read_csv(path_csv)

    if 'TIME' not in df.columns:
        raise ValueError(f"⚠️ Coluna 'TIME' não encontrada em {os.path.basename(path_csv)}")
    
    if df.shape[0] != 1440:
        raise ValueError(f"⚠️ Arquivo {os.path.basename(path_csv)} inválido: {df.shape[0]} linhas (esperado: 1440)")

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    if nome_coluna_energia not in df.columns:
        raise ValueError(f"⚠️ Coluna '{nome_coluna_energia}' não encontrada em {os.path.basename(path_csv)}")

    _local_cache[path_csv] = df
    return df

def processar_combinacao(idx_combinacao):
    idx, combinacao = idx_combinacao
    try:
        time.sleep(random.uniform(0, 0.02))

        dfs = []
        label_total = 0.0

        for path_csv in combinacao:
            df = carregar_csv(path_csv)
            df_sem_time = df.drop(columns=['TIME'])
            arr_flat = df_sem_time.values.flatten().astype(np.float32)
            dfs.append(arr_flat)
            label_total += df[nome_coluna_energia].iloc[-1]

        arr_final = np.concatenate(dfs).astype(np.float32)
        label_total_mes = label_total * 10

        nome_saida = f"amostra_{idx:04d}_{label_total_mes:.2f}.csv"
        path_saida = os.path.join(pasta_saida, nome_saida)
        tmp_path = path_saida + ".tmp"

        # Salvar com numpy para evitar MemoryError
        np.savetxt(tmp_path, [arr_final], delimiter=",", fmt="%.6f")
        os.replace(tmp_path, path_saida)

        return nome_saida

    except Exception as e:
        print(f"⚠️ Erro na combinação {idx}: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    arquivos_csv = sorted(glob.glob(f"{pasta_csv_diarios}/*.csv"))
    print(f"\n🔍 Encontrados {len(arquivos_csv)} arquivos de 1 dia")

    num_processos = min(16, mp.cpu_count())
    print(f"\n🚀 Usando {num_processos} processos paralelos\n")

    combinacoes_lista = list(combinations(arquivos_csv, 3))
    random.shuffle(combinacoes_lista)

    total_combinacoes = len(combinacoes_lista)
    print(f"🔢 Total de combinações possíveis: {total_combinacoes}")

    if N_amostras > total_combinacoes:
        N_amostras = total_combinacoes
        print(f"⚠️ Reduzido para {N_amostras} (máximo possível)")

    combinacoes_selecionadas = combinacoes_lista[:N_amostras]

    def enumerar_combinacoes():
        for idx, comb in enumerate(combinacoes_selecionadas):
            yield (idx, comb)

    pool = mp.Pool(processes=num_processos)
    nomes_arquivos_csv_gerados = list(tqdm(pool.imap_unordered(processar_combinacao, enumerar_combinacoes()), total=N_amostras))
    pool.close()
    pool.join()

    nomes_arquivos_csv_gerados = [x for x in nomes_arquivos_csv_gerados if x is not None]

    print(f"\n✅ Geração concluída: {len(nomes_arquivos_csv_gerados)} amostras OK")

    # Split train/test
    random.shuffle(nomes_arquivos_csv_gerados)
    N_test = int(0.2 * len(nomes_arquivos_csv_gerados))
    N_train = len(nomes_arquivos_csv_gerados) - N_test

    with open("train_files.txt", "w") as f_train, open("test_files.txt", "w") as f_test:
        for nome in nomes_arquivos_csv_gerados[:N_train]:
            f_train.write(nome + "\n")
        for nome in nomes_arquivos_csv_gerados[N_train:]:
            f_test.write(nome + "\n")

    print(f"\n📄 train_files.txt: {N_train} arquivos")
    print(f"📄 test_files.txt : {N_test} arquivos")
    print("\n🚀 Fim!")
