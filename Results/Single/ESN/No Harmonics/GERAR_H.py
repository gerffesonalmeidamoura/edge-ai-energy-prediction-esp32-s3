# GERAR_H_ESN.py
import re, pathlib, sys

SRC = "model_data.cc"       # ou um .h com o array
OUT = "model_data.h"
SYM = "g_model"

src = pathlib.Path(SRC).read_text(encoding="utf-8", errors="ignore")
m   = re.search(r'const\s+unsigned\s+char\s+(\w+)\[\]\s*=\s*\{([^}]*)\};', src, re.S)
if not m: raise SystemExit("Não encontrei array 'const unsigned char ...[] = {...};'")

name = m.group(1)
body = m.group(2)
# normaliza separadores
body = ",".join([x.strip() for x in body.replace("\n"," ").split(",") if x.strip()])

# tenta achar um comprimento, se existir no .cc
mlen = re.search(r'const\s+int\s+\w+_len\s*=\s*(\d+)\s*;', src)
length_define = f"const int {SYM}_len = {mlen.group(1)};" if mlen else ""

hdr = f"""#pragma once
#include <stdint.h>
#include <stddef.h>

// Gerado a partir de {SRC}
alignas(16) const unsigned char {SYM}[] = {{
  {body}
}};
{length_define if length_define else ""}
"""

# se não achou _len, calcula pela contagem de elementos
if not length_define:
    # conta inteiros/hex no body
    import re
    count = len(re.findall(r'0x[0-9A-Fa-f]{{2}}|\d+', body))
    hdr += f"const int {SYM}_len = {count};\n"

pathlib.Path(OUT).write_text(hdr, encoding="utf-8")
print(f"✅ Gerado {OUT}")
