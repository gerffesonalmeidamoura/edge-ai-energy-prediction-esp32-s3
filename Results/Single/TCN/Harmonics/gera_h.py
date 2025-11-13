import pathlib
b = pathlib.Path(r"C:\projeto_artigo\mono\TCN\com_harmonicas\modelo_final_quantizado\modelo_quant.tflite").read_bytes()
with open("model_data.h","w") as f:
    f.write('#pragma once\n#include <stdint.h>\n#include <stddef.h>\n')
    f.write('alignas(16) const unsigned char g_model[] = {')
    f.write(','.join(str(x) for x in b))
    f.write('};\nconst int g_model_len = ' + str(len(b)) + ';\n')
print("OK: model_data.h gerado")
