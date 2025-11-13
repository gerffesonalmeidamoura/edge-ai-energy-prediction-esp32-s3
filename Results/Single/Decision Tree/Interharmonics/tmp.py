import pickle, numpy as np

def load_pca(path="pca.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_get(obj, name):
    return getattr(obj, name, None)

pca = load_pca("pca.pkl")

# Lista o que o objeto realmente tem (pra debug rápido)
attrs = ["n_components", "n_components_", "n_samples_seen_", "components_",
         "explained_variance_ratio_", "explained_variance_", "var_",
         "singular_values_", "mean_", "whiten"]
have = {a: (safe_get(pca, a) is not None) for a in attrs}
print("ATTRS:", {k:v for k,v in have.items() if v})

# Quantos PCs efetivos
k = safe_get(pca, "n_components_") or safe_get(pca, "n_components")
if k is None:
    # tenta inferir por components_
    comps = safe_get(pca, "components_")
    k = comps.shape[0] if comps is not None else None
print(f"n_components_used = {k}")

# Caminho 1: já tem ratio pronto
evr = safe_get(pca, "explained_variance_ratio_")
if evr is not None:
    evr = np.asarray(evr, dtype=float)
    use = min(k, evr.shape[0]) if k is not None else evr.shape[0]
    evr_k = 100.0 * float(np.sum(evr[:use]))
    print(f"EVR_{use} = {evr_k:.2f}%  (from explained_variance_ratio_)")
    raise SystemExit(0)

# Caminho 2: calcular ratio = EV / sum(var_)
ev = safe_get(pca, "explained_variance_")
var = safe_get(pca, "var_")
if ev is not None and var is not None:
    ev = np.asarray(ev, dtype=float)
    total_var = float(np.sum(var))
    use = min(k, ev.shape[0]) if k is not None else ev.shape[0]
    evr_k = 100.0 * float(np.sum(ev[:use])) / total_var
    print(f"EVR_{use} = {evr_k:.2f}%  (from explained_variance_ / sum(var_))")
    raise SystemExit(0)

# Caminho 3: derivar EV pelos singulares e n_samples_seen_
sv = safe_get(pca, "singular_values_")
n_seen = safe_get(pca, "n_samples_seen_")
var = safe_get(pca, "var_")
if sv is not None and n_seen is not None and var is not None:
    ev = (np.asarray(sv, dtype=float)**2) / (float(n_seen) - 1.0)
    total_var = float(np.sum(var))
    use = min(k, ev.shape[0]) if k is not None else ev.shape[0]
    evr_k = 100.0 * float(np.sum(ev[:use])) / total_var
    print(f"EVR_{use} = {evr_k:.2f}%  (from singular_values_, n_samples_seen_, var_)")
    raise SystemExit(0)

# Se chegou aqui, faltam atributos essenciais
msg = [
 "Não foi possível calcular EVR porque o objeto não contém:",
 f"  - explained_variance_ratio_  -> {have.get('explained_variance_ratio_', False)}",
 f"  - explained_variance_        -> {have.get('explained_variance_', False)}",
 f"  - var_                       -> {have.get('var_', False)}",
 "Caminhos rápidos pra resolver:",
 "  (A) Instale a MESMA versão usada no treino: pip install -U 'scikit-learn==1.6.1'",
 "  (B) Ou recalcule o EVR diretamente dos dados (script abaixo em 'Opção 2')."
]
print("\n".join(msg))
