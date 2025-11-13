import os
import glob
import numpy as np
import shutil
from tqdm import tqdm

# Diretórios
pasta_origem = r"C:\projeto_artigo\trifasico\MLP\com_interharmonicas\teste"
pasta_ok = os.path.join(pasta_origem, "amostras_ok")
pasta_ruins = os.path.join(pasta_origem, "amostras_ruins")

os.makedirs(pasta_ok, exist_ok=True)
os.makedirs(pasta_ruins, exist_ok=True)

# Arquivos CSV
arquivos = sorted(glob.glob(os.path.join(pasta_origem, "amostra_*.csv")))
print(f"📂 {len(arquivos)} arquivos encontrados.")

# Verificação
for arq in tqdm(arquivos, desc="🔍 Verificando arquivos"):
    try:
        dados = np.loadtxt(arq, delimiter=",", dtype=np.float32)
        
        # Checar se tem NaN ou Inf
        if np.isnan(dados).any() or np.isinf(dados).any():
            shutil.move(arq, os.path.join(pasta_ruins, os.path.basename(arq)))
        else:
            shutil.move(arq, os.path.join(pasta_ok, os.path.basename(arq)))
    except Exception as e:
        print(f"⚠️ Erro ao ler {arq}: {e}")
        shutil.move(arq, os.path.join(pasta_ruins, os.path.basename(arq)))

print("\n✅ Separação concluída!")
print(f"Arquivos OK salvos em: {pasta_ok}")
print(f"Arquivos RUINS salvos em: {pasta_ruins}")
