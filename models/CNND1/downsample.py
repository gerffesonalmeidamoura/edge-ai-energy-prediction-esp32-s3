import os
import numpy as np
from scipy.signal import resample

# Diretórios corretos
pasta_entrada = r"C:\projeto_artigo\mono\MLP\com_harmonicas\dados_teste_3_dias"
pasta_saida = r"C:\projeto_artigo\mono\CNND1\com_harmonicas\convertidos_down_142560_teste"
os.makedirs(pasta_saida, exist_ok=True)

# Configuração
COLUNAS_FINAIS = 142560
arquivos = [f for f in os.listdir(pasta_entrada) if f.endswith(".csv")]

print(f"🔄 Processando {len(arquivos)} arquivos CSV (linha 1) com downsample para shape ({COLUNAS_FINAIS},)...")

for i, nome in enumerate(arquivos):
    caminho = os.path.join(pasta_entrada, nome)
    nome_base = os.path.splitext(nome)[0]
    try:
        with open(caminho, "r") as f:
            linhas = f.readlines()
            if len(linhas) < 2:
                raise ValueError("Arquivo com menos de 2 linhas")
            linha_dados = linhas[1].strip()  # ⚠️ Apenas a segunda linha (linha 1 real)
            dados = [float(x) for x in linha_dados.split(",") if x.strip()]

        if len(dados) < 1000:
            raise ValueError(f"Poucos dados válidos: {len(dados)}")

        # Aplicar downsample com scipy
        dados_resampled = resample(dados, COLUNAS_FINAIS)
        np.save(os.path.join(pasta_saida, f"{nome_base}.npy"), dados_resampled.astype(np.float32))

        if (i + 1) % 50 == 0 or i == len(arquivos) - 1:
            print(f"✅ {i+1}/{len(arquivos)} concluídos")

    except Exception as e:
        print(f"❌ Erro no arquivo {nome}: {e}")

print("🏁 Finalizado com sucesso.")
