import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─── Paths ─────────────────────────────────────────────────────────────────────
model_dir       = "modelo_final"
quant_dir       = "modelo_final_quantizado"
os.makedirs(quant_dir, exist_ok=True)

keras_model_fp  = os.path.join(model_dir, "modelo_final.keras")
x_pca_fp        = os.path.join(model_dir, "X_pca_final.npy")
tflite_out_fp   = os.path.join(quant_dir, "modelo_final_quantizado.tflite")

# ─── Load Keras model & calibration data ───────────────────────────────────────
print("📦 Loading Keras model...")
model = load_model(keras_model_fp)

print("🔍 Loading PCA-transformed samples for calibration...")
X_pca = np.load(x_pca_fp)

# ─── Representative dataset generator ─────────────────────────────────────────
def representative_dataset():
    for i in range(min(100, X_pca.shape[0])):
        # yield a batch of shape (1, n_components)
        yield [X_pca[i : i + 1].astype(np.float32)]

# ─── Convert to TFLite with post‑training INT8 quantization ────────────────────
print("⚙️ Converting to TFLite (INT8 quantization)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops   = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type        = tf.int8
converter.inference_output_type       = tf.int8

tflite_model = converter.convert()

# ─── Write out the quantized model ─────────────────────────────────────────────
with open(tflite_out_fp, "wb") as f:
    f.write(tflite_model)

print(f"\n✅ Quantized TFLite model saved to:\n   {tflite_out_fp}")
