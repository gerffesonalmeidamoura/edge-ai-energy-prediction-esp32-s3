import pandas as pd
import numpy as np

arquivo = r"C:\projeto_artigo\mono\MLP\com_interharmonicas\amostras_3dias_sem_harmonicas\amostra_0000_31.09.csv"

df = pd.read_csv(arquivo, skiprows=1)
df = df.iloc[:, 1:-1]  # Remover primeira e última colunas

print("🔎 Shape:", df.shape)
print("❓ Tem NaN?", df.isna().values.any())
print("⚠️ Total NaNs:", df.isna().sum().sum())
print("♾️ Tem Inf?", np.isinf(df.values).any())

colunas_constantes = df.columns[df.nunique() <= 1]
print("📌 Colunas constantes:", len(colunas_constantes))

colunas_monotonicas = [col for col in df.columns if df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing]
print("📈 Colunas monotônicas:", len(colunas_monotonicas))

# Exibir colunas suspeitas
print("🔍 Exemplo colunas constantes:", df[colunas_constantes].head(3))
print("🔍 Exemplo colunas monotônicas:", df[colunas_monotonicas[:3]].head(3))
