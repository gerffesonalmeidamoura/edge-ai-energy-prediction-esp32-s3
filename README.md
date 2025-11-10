
## Reproduction Steps

1. **Preprocessing**
   - Convert 1 Hz CSV logs into feature vectors (THD, harmonics, RMS, PF, energy, etc.)
   - Apply the saved `scaler.pkl` and optional PCA model to replicate training normalization

2. **Training**
   - Python scripts reproduce final CNN-1D, TCN-like, ESN and MLP models used in the paper
   - Output `.keras` models and quantized `.tflite` binaries

3. **On-device deployment**
   - Arduino sketch loads the `.tflite` model and performs inference on ESP32-S3
   - INA226 logs supply current and voltage traces to compute inference energy

4. **Energy & Latency Measurement**
   - Scripts in `scripts/` compute mean, best/worst, and 95% CIs
   - Supports baseline subtraction and arena watermark usage

## License
MIT License
