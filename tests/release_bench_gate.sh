#!/bin/sh
set -eu

model=${Q36_TEST_MODEL:-gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf}
prompt=${Q36_RELEASE_BENCH_PROMPT:-tests/long_context_story_prompt.txt}
min_prefill=${Q36_RELEASE_MIN_PREFILL_TPS:-50}
min_decode=${Q36_RELEASE_MIN_DECODE_TPS:-40}
out=$(mktemp /tmp/q36-release-bench.XXXXXX)
trap 'rm -f "$out"' EXIT INT TERM

./q36-bench -m "$model" --vulkan --prompt-file "$prompt" \
    --ctx-start 256 --ctx-max 256 --ctx-alloc 512 --gen-tokens 16 \
    --csv "$out"

awk -F, -v min_prefill="$min_prefill" -v min_decode="$min_decode" '
NR == 2 {
    printf "release benchmark: prefill %.2f t/s, decode %.2f t/s\n", $3, $5
    if ($3 + 0 < min_prefill) {
        printf "prefill gate failed: %.2f < %.2f t/s\n", $3, min_prefill > "/dev/stderr"
        exit 1
    }
    if ($5 + 0 < min_decode) {
        printf "decode gate failed: %.2f < %.2f t/s\n", $5, min_decode > "/dev/stderr"
        exit 1
    }
    found = 1
}
END {
    if (!found) {
        print "release benchmark produced no data row" > "/dev/stderr"
        exit 1
    }
}
' "$out"
