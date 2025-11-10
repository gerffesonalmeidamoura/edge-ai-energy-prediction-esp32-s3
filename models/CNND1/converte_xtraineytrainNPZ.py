import numpy as np

# Caminhos dos arquivos de entrada
arquivo_X = "X_train.npy"
arquivo_y = "y_train.npy"

# Nome do arquivo de saída
arquivo_saida = "dados_treino.npz"

# Carregar os dados
print("📥 Carregando arquivos .npy...")
X = np.load(arquivo_X)
y = np.load(arquivo_y)

print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")

# Salvar como .npz
np.savez(arquivo_saida, X=X, y=y)
print(f"💾 Arquivo .npz salvo como: {arquivo_saida}")
