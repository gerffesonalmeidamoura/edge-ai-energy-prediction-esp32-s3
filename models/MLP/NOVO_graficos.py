import json
import matplotlib.pyplot as plt

# Caminho para o JSON e saída
caminho_json = "modelo_final/training_history.json"
saida_arquivo = "modelo_final/training_validation_loss_kwh.png"

# Carregar histórico
with open(caminho_json, "r") as f:
    history = json.load(f)

# Verificar se as chaves estão presentes
if "loss" not in history or "val_loss" not in history:
    raise ValueError("Chaves 'loss' e/ou 'val_loss' não encontradas no training_history.json")

# Gerar gráfico
plt.figure(figsize=(10, 6))
plt.plot(history["loss"], label="Training Loss", marker='o', linewidth=1)
plt.plot(history["val_loss"], label="Validation Loss", marker='x', linewidth=1)
plt.xlabel("Epoch")
plt.ylabel("Loss (kWh)")
plt.title("Training vs Validation Loss (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(saida_arquivo, dpi=300)
plt.show()

print(f"✅ Gráfico salvo em: {saida_arquivo}")
