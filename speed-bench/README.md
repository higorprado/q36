## Benchmarking

Here we collect prefill and generation speed obtained with the BC-250.

Run `q36-bench` as:

```
./q36-bench \
  -m gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 \
  --ctx-max 32768 \
  --step-incr 2048 \
  --gen-tokens 128 \
  -ctk q8_0 -ctv q4_0
```

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/bc250.csv --title "BC-250 t/s"
```
