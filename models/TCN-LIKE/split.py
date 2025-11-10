import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# Caminhos
pasta_X = r"C:\projeto_artigo\mono\2_TCN\com_interharmonicas\convertidos_down_142560"
saida = r"C:\projeto_artigo\mono\2_TCN\com_interharmonicas"
os.makedirs(saida, exist_ok=True)

# Coletar arquivos .npy
arquivos = sorted([f for f in os.listdir(pasta_X) if f.endswith(".npy")])

X = []
y = []

for nome in arquivos:
    try:
        vetor = np.load(os.path.join(pasta_X, nome))
        if vetor.shape[0] != 142560:
            print(f"⚠️ Ignorado: {nome} (shape inválido: {vetor.shape})")
            continue
        if np.isnan(vetor).any() or np.isinf(vetor).any():
            print(f"❌ Ignorado (NaN/Inf): {nome}")
            continue
        match = re.search(r'_(\d+\.\d+)\.npy$', nome)
        if match:
            rotulo = float(match.group(1))  # consumo mensal
            X.append(vetor)
            y.append(rotulo)
        else:
            print(f"⚠️ Rótulo não encontrado no nome do arquivo: {nome}")
    except Exception as e:
        print(f"❌ Erro ao carregar {nome}: {e}")

# Reshape para CNN1D
X = np.array(X, dtype=np.float32).reshape((-1, 142560, 1))
y = np.array(y, dtype=np.float32)

# Criar faixas (bins) para estratificação
n_bins = 10  # número de faixas
est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
y_binned = est.fit_transform(y.reshape(-1, 1)).astype(int).flatten()

# Split com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y_binned
)

# Salvar
np.save(os.path.join(saida, "X_train.npy"), X_train)
np.save(os.path.join(saida, "X_test.npy"), X_test)
np.save(os.path.join(saida, "y_train.npy"), y_train)
np.save(os.path.join(saida, "y_test.npy"), y_test)

print("✅ Split e reshape concluídos com estratificação:")
print(f"  🔹 X_train: {X_train.shape}")
print(f"  🔹 X_test: {X_test.shape}")
