# AWQ Quantization — Solution

`quantize.sh` does the one-time AWQ quantization. Serve with `--quantization awq` in vLLM.

Typical: 50-75% memory reduction (14GB → 4GB for Mistral-7B), 1.3-1.7× faster inference, < 0.5pp MMLU drop.

Companion: [engineer-solutions/mod-110 ex-06](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-110-llm-infrastructure/exercise-06-inference-optimization-llm) for the full optimization chain.
