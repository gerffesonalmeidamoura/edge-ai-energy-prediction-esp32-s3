import pickle

with open("modelo_final_quantizado/pca.pkl", "rb") as f:
    pca = pickle.load(f)

print("Entradas esperadas pelo PCA:", pca.n_features_in_)
print("Saída do PCA (componentes):", pca.n_components)
