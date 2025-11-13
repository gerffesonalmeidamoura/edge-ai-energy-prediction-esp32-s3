# analisa_decision.py
import os, pickle, numpy as np
from pathlib import Path

# ===== CONFIG =====
ARQ_MODELO = "model_decision.pkl"   # árvore scikit-learn (regressor/classifier)
ARQ_PCA    = "pca.pkl"              # opcional: PCA do pré-processamento

def bytes_str(n):
    # retorna string com unidade (B/KB/MB)
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.2f} KB"
    return f"{n/1024**2:.2f} MB"

# ---------- Carrega árvore ----------
with open(ARQ_MODELO, "rb") as f:
    model = pickle.load(f)

is_classifier = hasattr(model, "classes_")
kind = "Classifier" if is_classifier else "Regressor"

# Acessa estrutura interna
tree = model.tree_
n_nodes   = tree.node_count
n_leaves  = model.get_n_leaves()
max_depth = model.get_depth()
n_feat_in = getattr(model, "n_features_in_", None)
criterion = getattr(model, "criterion", "n/a")

# Estimativas de “tamanho do modelo” em C/Flash:
#  - thresholds (float32) por nó interno
#  - features (int32) por nó
#  - filhos (left/right) int32 por nó
#  - valores finais (folhas) (float32) para regressor
n_internal = n_nodes - n_leaves
bytes_thresholds = n_internal * 4
bytes_features   = n_nodes   * 4
bytes_children   = n_nodes   * 2 * 4
if is_classifier:
    # valores por folha (probabilidades ou contagens) variam com n_classes_
    n_classes = len(model.classes_)
    bytes_values = n_leaves * n_classes * 4
else:
    bytes_values = n_leaves * 4

bytes_model_arrays = bytes_thresholds + bytes_features + bytes_children + bytes_values

# ---------- PCA (opcional) ----------
pca_info = ""
pca_bytes_flash = 0
if Path(ARQ_PCA).exists():
    with open(ARQ_PCA, "rb") as f:
        pca = pickle.load(f)
    n_comp = getattr(pca, "n_components_", None)
    n_in   = getattr(pca, "n_features_", None)
    evr    = getattr(pca, "explained_variance_ratio_", None)
    total_var = float(np.sum(evr)) if evr is not None else None
    # Tamanho típico a embarcar: mean (float32*n_in) + components (float32*n_comp*n_in)
    if (n_comp is not None) and (n_in is not None):
        pca_bytes_flash = (n_in * 4) + (n_comp * n_in * 4)
    pca_info = (
        f"\nPré-processamento (PCA)\n"
        f"Componentes: {n_comp} | Entrada original: {n_in}\n"
        + (f"Variância explicada (Σ): {total_var*100:.2f} %\n" if total_var is not None else "")
    )

# ---------- “INPUT/OUTPUT”, “MEMÓRIA”, “OPS” ----------
input_shape  = f"[1, {n_feat_in if n_feat_in is not None else 'K'}]"  # pós-PCA
output_shape = "[1]"
dtype = "float32"

# Memória de *execução* (não há arena): vetor de entrada + pequeno working set
# Entrada: K floats (p.ex. 50) → 50*4 = 200 B
# Working set ~ O(profundidade) → desprezível em relação à entrada
K = n_feat_in or 50
bytes_runtime = K * 4  # estimativa principal

# ---------- Print no leiaute “TFLite-like” ----------
print(f"📦 Tamanho do modelo (arrays em Flash, estimado): {bytes_str(bytes_model_arrays)}")
if pca_bytes_flash:
    print(f"📦 Tamanho do PCA (coeficientes em Flash, estimado): {bytes_str(pca_bytes_flash)}")

print("\n=== INPUT ===")
print(f"[0] name=features_pca shape={input_shape} dtype={dtype}")

print("\n=== OUTPUT ===")
print(f"[0] name=predict shape={output_shape} dtype={dtype}")

print(f"\n🔍 Memória de execução (estimativa): {bytes_str(bytes_runtime)}")
print(f"💡 Arena sugerida (TFLM): não se aplica (modelo em C, sem TFLM)")

print("\n🧩 \"Operações\" usadas (árvore de decisão)")
print(f"- Tipo: DecisionTree{kind}")
print(f"- Criterion: {criterion} | Nós: {n_nodes} | Folhas: {n_leaves} | Profundidade: {max_depth}")
print("- Núcleo computacional: comparações (x[feature] <= threshold) + desvio condicional até folha")
print("- Sem convoluções/ativação/filtros; sem quantização de tensores (FP32 nativo)")

if pca_info:
    print("\n" + pca_info)
