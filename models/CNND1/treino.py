import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Parâmetros
EPOCHS = 100
BATCH_SIZE = 2
INPUT_SHAPE = (142560, 1)
pasta_dados = r"C:\projeto_artigo\mono\CNND1\com_interharmonicas"
pasta_saida = os.path.join(pasta_dados, "resultado_treinamento_142560")
os.makedirs(pasta_saida, exist_ok=True)

# Carregar dados
X_train = np.load(os.path.join(pasta_dados, "X_train.npy"))
X_test = np.load(os.path.join(pasta_dados, "X_test.npy"))
y_train = np.load(os.path.join(pasta_dados, "y_train.npy"))
y_test = np.load(os.path.join(pasta_dados, "y_test.npy"))

print(f"📊 X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"📊 X_test: {X_test.shape}, y_test: {y_test.shape}")

# Criar modelo CNN1D
def criar_modelo(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Treinar modelo
modelo = criar_modelo(INPUT_SHAPE)
print("🚀 Iniciando treinamento...")
hist = modelo.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.2, verbose=1)

# Salvar modelo
modelo.save(os.path.join(pasta_saida, "modelo_final.keras"))
print("✅ Modelo salvo.")

# Plot - Loss
plt.figure()
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, "training_validation_loss.png"))

# Plot - MAE
plt.figure()
plt.plot(hist.history['mae'], label='Train MAE')
plt.plot(hist.history['val_mae'], label='Validation MAE')
plt.title("Training & Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, "training_validation_mae.png"))

# Predição final no teste
y_pred = modelo.predict(X_test, batch_size=2).flatten()

# Métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
erro_percentual = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Resultados
print("\n📊 Métricas no conjunto de teste:")
print(f"MAE: {mae:.2f} kWh")
print(f"RMSE: {rmse:.2f} kWh")
print(f"R²: {r2:.4f}")
print(f"Erro Percentual Médio: {erro_percentual:.2f}%")

# Salvar métricas
with open(os.path.join(pasta_saida, "resultados.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f} kWh\n")
    f.write(f"RMSE: {rmse:.2f} kWh\n")
    f.write(f"R²: {r2:.4f}\n")
    f.write(f"Erro Percentual Médio: {erro_percentual:.2f}%\n")

# Salvar CSV de resultados
df = pd.DataFrame({
    "real": y_test,
    "previsto": y_pred,
    "erro_abs": np.abs(y_test - y_pred),
    "erro_perc": np.abs((y_test - y_pred) / y_test) * 100
})
df.to_csv(os.path.join(pasta_saida, "resultados.csv"), index=False)
print("📁 Resultados salvos com sucesso.")
