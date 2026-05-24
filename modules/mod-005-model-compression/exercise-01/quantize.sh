#!/usr/bin/env bash
# AWQ int4 quantization for Mistral-7B.
python - <<'EOF'
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

src = "mistralai/Mistral-7B-Instruct-v0.2"
out = "mistral-7b-awq"
tok = AutoTokenizer.from_pretrained(src)
model = AutoAWQForCausalLM.from_pretrained(src)
model.quantize(tok, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})
model.save_quantized(out)
tok.save_pretrained(out)
print(f"saved {out}")
EOF
