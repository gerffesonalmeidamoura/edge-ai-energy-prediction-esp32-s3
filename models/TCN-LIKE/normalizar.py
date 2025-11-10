import os
import numpy as np

# Caminhos de entrada
base = r"C:\projeto_artigo\mono\CNND1\com_harmonicas"
arquivo_X_train = os.path.join(base, "X_train.npy")
arquivo_X_test = os.path.join(base, "X_test.npy")

# Carregar os dados
print("📥 Carregando dados...")
X_train = np.load(arquivo_X_train)
X_test = np.load(arquivo_X_test)
print(f"✅ X_train shape: {X_train.shape}")
print(f"✅ X_test shape: {X_test.shape}")

# Função de normalização Min-Max por amostra
def normalizar_por_amostra(X):
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)

# Aplicar normalização
print("🔄 Normalizando dados (MinMax por amostra)...")
X_train_norm = normalizar_por_amostra(X_train)
X_test_norm = normalizar_por_amostra(X_test)

# Salvar os novos arquivos
np.save(os.path.join(base, "X_train_norm.npy"), X_train_norm)
np.save(os.path.join(base, "X_test_norm.npy"), X_test_norm)
print("✅ Arquivos normalizados salvos com sucesso!")
