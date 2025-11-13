# tflite_to_h.py
import numpy as np, sys
inp = sys.argv[1]  # ex: modelo_quant.tflite
data = np.fromfile(inp, dtype=np.uint8)
with open(inp.rsplit('.',1)[0] + ".h", "w") as f:
    f.write("// auto-generated from %s\n" % inp)
    f.write("const unsigned char g_model[] = {")
    for i,b in enumerate(data):
        f.write(str(int(b)) + ("," if i<len(data)-1 else ""))
        if (i+1)%32==0: f.write("\n")
    f.write("};\nconst int g_model_len = %d;\n" % len(data))
print("OK")
