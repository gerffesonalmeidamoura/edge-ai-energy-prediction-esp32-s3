# ANALISA_ESN.py
import os, re, numpy as np, tensorflow as tf

CAMINHO = "model_data.cc"  # ou "modelo_esn_int8.tflite"

def extract_from_cc(path_cc: str) -> bytes:
    src = open(path_cc, "r", encoding="utf-8", errors="ignore").read()
    m = re.search(r'const\s+unsigned\s+char\s+\w+\[\]\s*=\s*\{([^}]*)\};', src, re.S)
    if not m: raise RuntimeError("Não achei array de bytes no .cc/.h")
    body = m.group(1)
    # aceita '0x..' OU inteiros decimais
    bytes_list = []
    for tok in re.findall(r'0x[0-9A-Fa-f]{2}|\d+', body):
        if tok.startswith("0x"): bytes_list.append(int(tok,16))
        else: bytes_list.append(int(tok))
    return bytes(bytes_list)

def load_model_bytes(path: str) -> bytes:
    if path.lower().endswith((".cc",".h",".hpp",".c")):
        return extract_from_cc(path)
    return open(path,"rb").read()

buf = load_model_bytes(CAMINHO)
print(f"📦 Tamanho do modelo: {len(buf)/1024:.2f} KB")

# Interpreter a partir de bytes
interpreter = tf.lite.Interpreter(model_content=buf)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()
out = interpreter.get_output_details()

print("\n=== INPUT ===")
for i,d in enumerate(inp):
    print(f"[{i}] name={d['name']} shape={d['shape']} dtype={d['dtype']}")
    if 'quantization' in d and d['quantization'][0] > 0:
        s,z = d['quantization']; print(f"     quant(scale={s}, zero_point={z})")

print("\n=== OUTPUT ===")
for i,d in enumerate(out):
    print(f"[{i}] name={d['name']} shape={d['shape']} dtype={d['dtype']}")
    if 'quantization' in d and d['quantization'][0] > 0:
        s,z = d['quantization']; print(f"     quant(scale={s}, zero_point={z})")

# estimativa “arena”
td = interpreter.get_tensor_details()
tot = 0
for t in td:
    shape = t['shape'] if 'shape' in t else [1]
    dtype = t['dtype'] if 'dtype' in t else np.float32
    tot += int(np.prod(shape)) * np.dtype(dtype).itemsize
print(f"\n🔍 Estimativa de memória total dos tensores: {tot/1024:.2f} KB")
print(f"💡 Arena sugerida (TFLM): {(tot*2 + 65536)/1024:.2f} KB")
