# gerar_model_data_dt.py
import re, sys, pathlib

IN  = "modelo_decision_inferencia.c"
OUT = "model_data.h"
SYMBOL = "dt_predict"   # nome público

src = pathlib.Path(IN).read_text(encoding="utf-8", errors="ignore")

# 1) Deduza a dimensionalidade: maior índice f[XXX] + 1
idxs = [int(m.group(1)) for m in re.finditer(r'\bf\[(\d+)\]', src)]
dim  = (max(idxs) + 1) if idxs else 50  # fallback seguro

# 2) Renomeie a função predict -> dt_predict e torne header-only
#    Suporta "float predict(" com ou sem "static"
src2 = re.sub(r'\b(float\s+)(?:static\s+)?predict\s*\(',
              r'static inline \1' + SYMBOL + '(', src, count=1)

# 3) Construa o header
header = []
header.append("#pragma once")
header.append("#include <stdint.h>")
header.append("#include <stddef.h>")
header.append("")
header.append("// ==== Decision Tree (header-only) ====")
header.append(f"#define DT_INPUT_DIM {dim}")
header.append("")
header.append("// Assinatura pública:")
header.append(f"//   float {SYMBOL}(const float f[DT_INPUT_DIM]);")
header.append("")
header.append("// Implementação gerada a partir de modelo_decision_inferencia.c")
header.append("// Torna a função inline para uso direto no firmware.")
header.append("")
header.append(src2)
header.append("")

pathlib.Path(OUT).write_text("\n".join(header), encoding="utf-8")
print(f"✅ Gerado {OUT} (DT_INPUT_DIM={dim}, função={SYMBOL})")
