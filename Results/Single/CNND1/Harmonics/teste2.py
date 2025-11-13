from tensorflow.lite.experimental import Analyzer
print(Analyzer.analyze(model_path="modelo_quant.tflite"))
