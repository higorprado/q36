# Q36 GGUF Tools

Tools for building the Q36 mixed quant of `Qwen3.6-35B-A3B` from:

```text
gguf/Qwen3.6-35B-A3B-Q8_0.gguf
```

The default recipe is:

- `blk.N.ffn_gate_exps.weight` -> `IQ2_XXS`
- `blk.N.ffn_up_exps.weight` -> `IQ2_XXS`
- `blk.N.ffn_down_exps.weight` -> `Q2_K`
- every other tensor stays at the source type, normally `Q8_0`

The quantizer can also promote a routed expert layer window to `Q4_K`.
For the Antirez last-six expert window, use `--q4-expert-last 6`.
On the local Qwen3.6 model this resolves to layers `34..39`.

Shared experts, router/gating tensors, embeddings, output heads, norms,
attention, SSM tensors, and other non-routed tensors are explicit keep-list
entries in `qwen36-quantize.c`. Unknown tensor names fail the dry-run.

## Build

```sh
make -C gguf-tools
```

`qwen36-quantize` is plain C. It mmaps the input GGUF and decodes converted
tensors in small chunks before writing their mixed quantized form.

BF16 and F16 inputs are decoded in the same bounded chunks. Repeat `--in` in
shard order for split models; tensors remain mmap-backed in their source shard.
Keep-list BF16/F16 tensors become Q8_0, while F32 tensors remain F32, producing
the same target type layout as the Q8-source recipe.

```sh
gguf-tools/qwen36-quantize \
  --in gguf/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf \
  --in gguf/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf \
  --imatrix gguf/Qwen3.6-35B-A3B-chat-v2-routed-moe-q36.dat \
  --imatrix-strict \
  --threads 12 \
  --out gguf/Qwen3.6-35B-A3B-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-v2-imatrix-bf16.gguf
```

## Workflow

1. Start from the local Q8 GGUF:

```text
gguf/Qwen3.6-35B-A3B-Q8_0.gguf
```

2. Build the JSONL calibration corpus:

```sh
python3 gguf-tools/imatrix/dataset/build_q36_imatrix_dataset.py
```

3. Render the corpus with the Qwen3.6 chat template:

```sh
python3 gguf-tools/imatrix/dataset/render_q36_prompts.py \
  --in gguf-tools/imatrix/dataset/prompts.jsonl
```

This writes `rendered_prompts.txt`, `rendered_prompts_think.txt`, and
`rendered_prompts_nothink.txt`. Use the combined file for normal calibration.

4. Collect imatrix statistics locally with llama.cpp and mmap:

```sh
llama.cpp/build/bin/llama-imatrix \
  -m gguf/Qwen3.6-35B-A3B-Q8_0.gguf \
  -f gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --mmap \
  --parse-special \
  --output-format dat \
  -o gguf/q36-qwen36-imatrix.dat
```

5. Dry-run and audit the quantization plan:

```sh
gguf-tools/qwen36-quantize \
  --in gguf/Qwen3.6-35B-A3B-Q8_0.gguf \
  --imatrix gguf/q36-qwen36-imatrix.dat \
  --imatrix-strict \
  --audit gguf/q36-mixed-audit.tsv \
  --dry-run
```

Expected Q36 plan: 80 gate/up routed tensors to `IQ2_XXS`, 40 down routed
tensors to `Q2_K`, and all remaining tensors kept Q8.

For the last-six Q4 expert variant:

```sh
gguf-tools/qwen36-quantize \
  --in gguf/Qwen3.6-35B-A3B-Q8_0.gguf \
  --imatrix gguf/imatrix_unsloth.gguf_file \
  --imatrix-strict \
  --q4-expert-last 6 \
  --audit gguf/q36-unsloth-last6-q4-quant-audit.tsv \
  --dry-run
```

Expected plan: layers `34..39`, 68 gate/up routed tensors to `IQ2_XXS`,
34 down routed tensors to `Q2_K`, 18 routed tensors to `Q4_K`, and all
remaining tensors kept at their source type.

6. Write the mixed GGUF:

```sh
gguf-tools/qwen36-quantize \
  --in gguf/Qwen3.6-35B-A3B-Q8_0.gguf \
  --imatrix gguf/q36-qwen36-imatrix.dat \
  --imatrix-strict \
  --audit gguf/q36-mixed-audit.tsv \
  --out gguf/Qwen3.6-35B-A3B-Q36-IQ2XXS-gateup-Q2K-down-Q8rest.gguf
```

Or write the last-six Q4 expert variant:

```sh
gguf-tools/qwen36-quantize \
  --in gguf/Qwen3.6-35B-A3B-Q8_0.gguf \
  --imatrix gguf/imatrix_unsloth.gguf_file \
  --imatrix-strict \
  --q4-expert-last 6 \
  --audit gguf/q36-unsloth-last6-q4-quant-audit.tsv \
  --out gguf/Qwen3.6-35B-A3B-Layers34-39Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-unsloth-imatrix.gguf
```

7. Run quality testing with the OpenRouter reference set and the native Q36
scorer. See `gguf-tools/quality-testing/README.md`.
