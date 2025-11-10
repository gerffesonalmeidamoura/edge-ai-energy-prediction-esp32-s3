#!/usr/bin/env python3
# csv_to_npy_fixed.py

import os
import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

# ─── CONFIGURAÇÃO ───────────────────────────────────────────────────────────────
INPUT_DIR   = r"C:\projeto_artigo\mono\MLP\com_interharmonicas\teste"
OUTPUT_DIR  = r"C:\projeto_artigo\mono\CNND1\com_interharmonicas\teste_npy"
TARGET_LEN  = 142560  # comprimento fixo desejado

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── CAPTURA TODOS OS CSVs ──────────────────────────────────────────────────────
csv_files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv"))
print(f"🔍 Encontrados {len(csv_files)} CSVs em '{INPUT_DIR}'\n")

# ─── PROCESSAMENTO ──────────────────────────────────────────────────────────────
for fname in tqdm(csv_files, desc="Convertendo CSV → NPY"):
    src = os.path.join(INPUT_DIR, fname)
    try:
        # 1) Leitura (mantém cabeçalho)
        df = pd.read_csv(src)

        # 2) Remove colunas irrelevantes
        #    - Qualquer "Unnamed" (NaNs)
        #    - A coluna de tempo (timestamp)
        #    - A coluna de energia acumulada (rótulo)
        drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
        for c in ["timestamp", "EnergiaAcumulada_KWh"]:
            if c in df.columns:
                drop_cols.append(c)
        df = df.drop(columns=drop_cols, errors="ignore")

        # 3) Converte tudo para numérico (coerce → NaN) e elimina colunas vazias
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

        # 4) Achata em 1D
        arr = df.values.flatten()

        # 5) Valida NaN/Inf
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"❌ Ignorado (NaN/Inf): {fname}")
            continue

        # 6) Se o vetor não tiver TARGET_LEN, resample para corrigi‑lo
        if arr.size != TARGET_LEN:
            arr = resample(arr, TARGET_LEN).astype(np.float32)

        # 7) Salva .npy com shape exato (TARGET_LEN,)
        base = os.path.splitext(fname)[0]
        out_fp = os.path.join(OUTPUT_DIR, base + ".npy")
        np.save(out_fp, arr.astype(np.float32))

    except Exception as e:
        print(f"❌ Erro ao processar {fname}: {e}")

print("\n✅ Conversão concluída. .npy gerados em:")
print("   ", OUTPUT_DIR)
