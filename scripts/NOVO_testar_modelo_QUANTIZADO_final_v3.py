import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import warnings

# Para suprimir avisos de RankWarning no polyfit
warnings.simplefilter("ignore", np.RankWarning)

# Pastas
pasta_modelo = "modelo_final_quantizado"
pasta_amostras = "teste"
os.makedirs("resultados_teste_quantizado", exist_ok=True)

# Carregar modelo TFLite
print(f"\n📦 Carregando modelo quantizado: {pasta_modelo}/modelo_final_quantizado.tflite")
interpreter = tf.lite.Interpreter(model_path=f"{pasta_modelo}/modelo_final_quantizado.tflite")
interpreter.allocate_tensors()

# Detalhes dos tensores
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']

print(f"🧠 Modelo TFLite carregado com entrada {input_dtype} e saída {output_dtype}")

# Carregar PCA
with open(f"{pasta_modelo}/pca.pkl", "rb") as f:
    pca = pickle.load(f)

print(f"🔧 PCA espera {pca.n_features_in_} variáveis de entrada")

# Amostras
arquivos_csv = sorted(glob.glob(f"{pasta_amostras}/*.csv"))
print(f"\n🔍 Lendo {len(arquivos_csv)} amostras de teste...\n")

# Resultados
resultados = []

for path in tqdm(arquivos_csv, desc="Testando amostras"):
    try:
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = arr.flatten().reshape(1, -1)

        if arr.shape[1] != pca.n_features_in_:
            print(f"⚠️ Shape mismatch em {os.path.basename(path)}: {arr.shape[1]} colunas vs {pca.n_features_in_} esperadas.")
            continue

        # Redução via PCA
        X_pca = pca.transform(arr).astype(np.float32)

        # Inferência TFLite
        interpreter.set_tensor(input_details[0]['index'], X_pca)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        y_pred = float(output.flatten()[0])

        # Nome e valor real
        nome = os.path.basename(path)
        valor_real = float(nome.split("_")[-1].replace(".csv", ""))

        resultados.append((nome, valor_real, y_pred))

        # Print de debug
        print(f"{nome}: real={valor_real:.2f} | predito={y_pred:.2f}")

    except Exception as e:
        print(f"⚠️ Erro em {os.path.basename(path)}: {e}")

# Criar DataFrame
df_res = pd.DataFrame(resultados, columns=["amostra", "valor_real", "valor_predito"])
df_res.to_csv("resultados_teste_quantizado/resultados.csv", index=False)

# Correção polinomial grau 2
try:
    coef2, coef1, coef0 = np.polyfit(df_res["valor_predito"], df_res["valor_real"], deg=2)
    df_res["valor_predito_corrigido"] = coef2 * df_res["valor_predito"]**2 + coef1 * df_res["valor_predito"] + coef0

    print(f"\n📈 Correção polinomial aplicada: y = ({coef2:.6f})*x² + ({coef1:.6f})*x + ({coef0:.6f})")
except Exception as e:
    print(f"❌ Erro ao aplicar correção polinomial: {e}")
    df_res["valor_predito_corrigido"] = df_res["valor_predito"]

df_res.to_csv("resultados_teste_quantizado/resultados_corrigidos.csv", index=False)

# Métricas
mae = np.mean(np.abs(df_res["valor_real"] - df_res["valor_predito_corrigido"]))
rmse = np.sqrt(np.mean((df_res["valor_real"] - df_res["valor_predito_corrigido"]) ** 2))
ss_res = np.sum((df_res["valor_real"] - df_res["valor_predito_corrigido"]) ** 2)
ss_tot = np.sum((df_res["valor_real"] - np.mean(df_res["valor_real"])) ** 2)
r2 = 1 - (ss_res / ss_tot)

with open("resultados_teste_quantizado/metrics_test.txt", "w") as f:
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

print(f"\n✅ Resultados salvos em:")
print("   → resultados_teste_quantizado/resultados.csv")
print("   → resultados_teste_quantizado/resultados_corrigidos.csv")
print(f"\n🎯 Final TEST metrics (TFLite): MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
