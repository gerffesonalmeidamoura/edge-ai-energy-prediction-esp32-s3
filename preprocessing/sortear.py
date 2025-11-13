import os
import glob
import random
import shutil

# Pasta de origem
pasta_origem = "dados_teste_3_dias"
# Pasta de destino
pasta_destino = os.path.join(pasta_origem, "teste")
os.makedirs(pasta_destino, exist_ok=True)

# Quantas amostras você quer sortear
n_amostras = 266

# Encontrar todas as amostras
arquivos_csv = sorted(glob.glob(os.path.join(pasta_origem, "amostra_*.csv")))

print(f"\n🔍 Encontradas {len(arquivos_csv)} amostras totais")

# Garantir que tem amostras suficientes
if len(arquivos_csv) < n_amostras:
    raise ValueError(f"⚠️ Apenas {len(arquivos_csv)} amostras disponíveis, não dá para sortear {n_amostras}")

# Sortear aleatoriamente
amostras_sorteadas = random.sample(arquivos_csv, n_amostras)

# Mover amostras sorteadas
for i, path in enumerate(amostras_sorteadas):
    nome_arquivo = os.path.basename(path)
    destino = os.path.join(pasta_destino, nome_arquivo)
    shutil.move(path, destino)

print(f"\n✅ {n_amostras} amostras MOVIDAS para '{pasta_destino}'\n🎉 Fim!")
