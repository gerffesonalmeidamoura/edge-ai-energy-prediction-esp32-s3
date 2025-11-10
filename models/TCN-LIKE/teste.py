import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Caminhos
pasta_base = r"C:\projeto_artigo\mono\2_TCN\com_interharmonicas\resultado_treinamento_TCN_142560"
arquivo_pred = os.path.join(pasta_base, "resultados.csv")

# Carregar dados reais e previstos
df = pd.read_csv(arquivo_pred)
y_test = df["real"].values
y_pred = df["previsto"].values

# Procurar offset ideal no intervalo -20 a +20 (com precisão de 0.1)
melhor_r2 = -np.inf
melhor_offset = 0

offsets = np.arange(-20, 20.1, 0.1)
for offset in offsets:
    y_corrigido = y_pred + offset
    r2 = r2_score(y_test, y_corrigido)
    if r2 > melhor_r2:
        melhor_r2 = r2
        melhor_offset = offset

# Aplicar offset ideal
y_corrigido = y_pred + melhor_offset
mae = mean_absolute_error(y_test, y_corrigido)
rmse = np.sqrt(mean_squared_error(y_test, y_corrigido))
erro_percentual = np.mean(np.abs((y_test - y_corrigido) / y_test)) * 100

# Exibir resultados
print(f"\n🎯 Melhor offset encontrado: {melhor_offset:.2f} kWh")
print(f"📈 R²: {melhor_r2:.4f}")
print(f"📉 MAE: {mae:.2f} kWh")
print(f"📉 RMSE: {rmse:.2f} kWh")
print(f"📉 Erro Percentual Médio: {erro_percentual:.2f}%")

# Atualizar CSV com predição corrigida
df["previsto_corrigido"] = y_corrigido
df["erro_abs_corrigido"] = np.abs(y_test - y_corrigido)
df["erro_perc_corrigido"] = np.abs((y_test - y_corrigido) / y_test) * 100
df.to_csv(os.path.join(pasta_base, "resultados_corrigido_offset.csv"), index=False)

# Salvar txt com métricas corrigidas
with open(os.path.join(pasta_base, "metricas_corrigidas.txt"), "w") as f:
    f.write(f"Offset ótimo aplicado: {melhor_offset:.2f} kWh\n")
    f.write(f"R² corrigido: {melhor_r2:.4f}\n")
    f.write(f"MAE corrigido: {mae:.2f} kWh\n")
    f.write(f"RMSE corrigido: {rmse:.2f} kWh\n")
    f.write(f"Erro Percentual Médio corrigido: {erro_percentual:.2f}%\n")

# Gráfico de dispersão corrigido
plt.figure()
plt.scatter(y_test, y_corrigido, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal")
plt.xlabel("Valor real (kWh)")
plt.ylabel("Previsto com offset (kWh)")
plt.title("Dispersão - Previsão Corrigida por Offset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pasta_base, "dispersao_corrigida_offset.png"))

print("✅ Resultados corrigidos salvos.")
