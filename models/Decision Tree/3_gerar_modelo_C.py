import os
import pickle
from sklearn.tree import _tree

# Caminho do modelo
pasta_modelo = r"C:\projeto_artigo\mono\2_DECISION\2_com_interharmonicas\saida_pca_decision"
modelo_path = os.path.join(pasta_modelo, "model_decision.pkl")
saida_c = os.path.join(pasta_modelo, "modelo_decision_inferencia.c")

# Carregar o modelo Decision Tree
with open(modelo_path, "rb") as f:
    model = pickle.load(f)

# Função para converter a árvore para C
def export_tree_to_c(tree, feature_names=None):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if feature_names else f"f[{i}]"
        for i in tree_.feature
    ]

    def recurse(node, depth=1):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            return (
                f"{indent}if ({feature_name[node]} <= {tree_.threshold[node]:.6f}) {{\n"
                f"{recurse(tree_.children_left[node], depth + 1)}"
                f"{indent}}} else {{\n"
                f"{recurse(tree_.children_right[node], depth + 1)}"
                f"{indent}}}\n"
            )
        else:
            return f"{indent}return {tree_.value[node][0][0]:.6f};\n"

    body = recurse(0)
    c_code = (
        "float predict(float f[]) {\n"
        f"{body}"
        "}\n"
    )
    return c_code

# Gerar e salvar o código C
codigo_c = export_tree_to_c(model)

with open(saida_c, "w") as f:
    f.write(codigo_c)

print(f"✅ Modelo C gerado em: {saida_c}")
