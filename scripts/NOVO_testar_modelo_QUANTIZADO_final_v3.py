import os
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from tqdm import tqdm

# Suprimir RankWarning do polyfit
warnings.simplefilter("ignore", np.RankWarning)

# Pastas
pasta_modelo   = "modelo_quantizado"
pasta_amostras = "teste"
pasta_saida    = "resultados_teste_quantizado"
os.makedirs(pasta_saida, exist_ok=True)

# Parâmetros
sample_step = 10  # mesmo downsampling usado no treinamento

# Carregar modelo TFLite
print(f"\n📦 Carregando TFLite: {pasta_modelo}/modelo_quantizado.tflite")
interpreter = tf.lite.Interpreter(model_path=os.path.join(pasta_modelo, "modelo_quantizado.tflite"))
interpreter.allocate_tensors()
input_details   = interpreter.get_input_details()[0]
output_details  = interpreter.get_output_details()[0]
in_dtype        = input_details['dtype']
in_scale, in_zero_point   = input_details['quantization']
out_scale, out_zero_point = output_details['quantization']

# Carregar PCA
with open(os.path.join(pasta_modelo, "pca.pkl"), "rb") as f:
    pca = pickle.load(f)
print(f"🔧 PCA espera {pca.n_features_in_} features")

# Listar amostras
csvs = sorted(glob.glob(os.path.join(pasta_amostras, "*.csv")))
print(f"\n🔍 Encontradas {len(csvs)} amostras em '{pasta_amostras}'\n")

resultados = []

for path in tqdm(csvs, desc="Inferindo (TFLite)"):
    nome = os.path.basename(path)
    try:
        # Leitura igual ao treinamento
        df = pd.read_csv(path)
        df = df.drop(df.columns[0], axis=1)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df.fillna(0.0)
        df = df.iloc[::sample_step, :].reset_index(drop=True)
        arr = df.to_numpy(dtype="float32").flatten().reshape(1, -1)

        # Verificar compatibilidade
        if arr.shape[1] != pca.n_features_in_:
            print(f"⚠️ Shape mismatch em {nome}: {arr.shape[1]} ≠ {pca.n_features_in_}")
            continue

        # PCA
        X_pca = pca.transform(arr).astype("float32")

        # Quantizar para INT8
        quant_input = np.clip(
            np.round(X_pca / in_scale + in_zero_point),
            np.iinfo(in_dtype).min, np.iinfo(in_dtype).max
        ).astype(in_dtype)
        interpreter.set_tensor(input_details['index'], quant_input)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details['index']).astype(np.int32)

        # Dequantizar saída
        y_pred = float((raw_output - out_zero_point) * out_scale)

        # Valor real
        valor_real = float(nome.split("_")[-1].replace(".csv", ""))
        resultados.append((nome, valor_real, y_pred))

    except Exception as e:
        print(f"⚠️ Erro em {nome}: {e}")

# DataFrame de resultados
df_res = pd.DataFrame(resultados, columns=["amostra", "valor_real", "valor_predito"])
df_res.to_csv(os.path.join(pasta_saida, "resultados.csv"), index=False)

# Correção polinomial grau 2
if len(df_res) > 1:
    coef2, coef1, coef0 = np.polyfit(df_res["valor_predito"], df_res["valor_real"], deg=2)
    df_res["valor_predito_corrigido"] = (
        coef2 * df_res["valor_predito"]**2 +
        coef1 * df_res["valor_predito"] +
        coef0
    )
    print(f"\n📈 Correção polinomial: y = {coef2:.6f}x² + {coef1:.6f}x + {coef0:.6f}")
else:
    df_res["valor_predito_corrigido"] = df_res["valor_predito"]
    print("❌ Sem dados suficientes para ajuste polinomial.")

df_res.to_csv(os.path.join(pasta_saida, "resultados_corrigidos.csv"), index=False)

# Métricas finais
if len(df_res) > 0:
    mae  = np.mean(np.abs(df_res["valor_real"] - df_res["valor_predito_corrigido"]))
    rmse = np.sqrt(np.mean((df_res["valor_real"] - df_res["valor_predito_corrigido"])**2))
    ss_res = np.sum((df_res["valor_real"] - df_res["valor_predito_corrigido"])**2)
    ss_tot = np.sum((df_res["valor_real"] - df_res["valor_real"].mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else float("nan")
else:
    mae = rmse = r2 = float("nan")

with open(os.path.join(pasta_saida, "metrics_test.txt"), "w") as f:
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

print(f"\n✅ Resultados salvos em '{pasta_saida}/':")
print("   • resultados.csv")
print("   • resultados_corrigidos.csv")
print(f"\n🎯 Métricas TFLite: MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
