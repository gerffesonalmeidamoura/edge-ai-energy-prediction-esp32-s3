On-Device Edge AI for Monthly Energy Forecasting on ESP32-S3 Full
Reproduction Guide and Project Structure

This repository contains all datasets, processing pipelines, firmware,
and experimental artifacts used in the paper: “On-Device Edge AI for
Monthly Energy Forecasting on ESP32-S3 Using 3-Day Windows”

It supports full reproducibility for: • Data preparation • Model
training • Quantization • Inference • ESP32-S3 energy/memory/latency
profiling • Visualization of all metrics and figures

  --------------------
  Directory Overview
  --------------------

Firmware/ Contains all embedded code for the ESP32-S3: - ADE7880 +
ESP32-S3 measurement firmware - INA226 power-measurement firmware -
Per-model benchmarking sketches (latency, PSRAM, Flash, energy) - Code
used to reproduce device-side results

Results/ Contains all results for the 30 evaluated models: - Efficiency/
: Memory, latency, energy per inference - Metrics/ : R², MAE, RMSE -
Single/ : Single-phase figures, predictions, CSVs - Three/ : Three-phase
figures, predictions, CSVs

Samples/ Contains the real samples used to train the models: - Original
Data/ separated into: • No Harmonics • Harmonics • Interharmonics

preprocessing/ Includes: - PCA computation - Downsampling & filtering -
Dataset generation - Train/test splitting

scripts/ All Python scripts for: - Data generation - Training -
Quantization - Inference - Metric extraction - Visualization

  --------------------
  Reproduction Steps
  --------------------

1.  Generate 3-day combinations Run: scripts/gerar_3_dias_V2.py

2.  Random train/test split Run: scripts/sortear.py

3.  Optional PCA computation Run:
    scripts/NOVO_treinamento_pca_global_v4.py

4.  Train the models Run: scripts/NOVO_treinamento_pca_global_v5.py
    Output: model_final.keras, logs, normalization stats

5.  Float inference Run: scripts/NOVO_testar_modelo_final_v3.py

    Extract metrics: scripts/extrair_resultados_COMPLETO_V2.py

6.  Train the quantized model Run:
    scripts/NOVO_treinamento_QUANTIZADO_v3_OK.py

7.  Quantized inference Run:
    scripts/NOVO_testar_modelo_QUANTIZADO_final_v3.py

    Extract metrics:
    scripts/extrair_resultados_QUANTIZADO_COMPLETO_V2.py

8.  ESP32-S3 on-device profiling Steps:

    -   Convert .tflite → model_data.h
    -   Flash firmware from Firmware/
    -   Run inference loops
    -   Collect INA226 logs
    -   Compute: • latency (µs) • energy per inference (µJ) • Flash &
        PSRAM usage

  ---------
  License
  ---------

MIT License
