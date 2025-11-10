import os
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# === CONFIGURAÇÕES ===
pasta_base = r"C:\projeto_artigo\mono\2_DECISION\2_com_interharmonicas\saida_pca_decision"
arquivo_X = os.path.join(pasta_base, "X_pca.npy")
arquivo_y = os.path.join(pasta_base, "y.npy")
saida_modelo = os.path.join(pasta_base, "model_decision.pkl")
saida_csv = os.path.join(pasta_base, "resultados_decision.csv")
saida_metricas = os.path.join(pasta_base, "metricas_decision.txt")

# === CRIA DIRETÓRIO DE SAÍDA ===
os.makedirs(pasta_base, exist_ok=True)

# === CARREGAR DADOS ===
X = np.load(arquivo_X)
y = np.load(arquivo_y)

# === DIVISÃO EM TREINO E TESTE ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"📊 X_test: {X_test.shape}, y_test: {y_test.shape}")

# === TREINAMENTO ===
print("🌳 Treinando Decision Tree Regressor...")
start = time.time()
model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, y_train)
end = time.time()
print(f"✅ Treinamento concluído em {end - start:.2f} segundos.")

# === SALVAR MODELO ===
with open(saida_modelo, "wb") as f:
    pickle.dump(model, f)
print(f"🧠 Modelo Decision Tree salvo com sucesso em: {saida_modelo}")

# === PREDIÇÃO E AVALIAÇÃO ===
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
erro_percentual = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# === EXIBIR MÉTRICAS ===
print("\n📊 Métricas no conjunto de teste:")
print(f"MAE: {mae:.2f} kWh")
print(f"RMSE: {rmse:.2f} kWh")
print(f"R²: {r2:.4f}")
print(f"Erro Percentual Médio: {erro_percentual:.2f}%")

# === SALVAR RESULTADOS CSV ===
df = pd.DataFrame({
    "real": y_test,
    "previsto": y_pred,
    "erro_abs": np.abs(y_test - y_pred),
    "erro_perc": np.abs((y_test - y_pred) / y_test) * 100
})
df.to_csv(saida_csv, index=False)

# === SALVAR MÉTRICAS TXT ===
with open(saida_metricas, "w") as f:
    f.write(f"MAE: {mae:.2f} kWh\n")
    f.write(f"RMSE: {rmse:.2f} kWh\n")
    f.write(f"R²: {r2:.4f}\n")
    f.write(f"Erro Percentual Médio: {erro_percentual:.2f}%\n")

print(f"📁 Resultados salvos em:\n- {saida_modelo}\n- {saida_csv}\n- {saida_metricas}")
