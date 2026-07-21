#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "llama.h"

struct token_prob {
    llama_token id;
    float logit;
    float logprob;
};

static std::string read_file(const char *path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("failed to read ") + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20 || c >= 0x80) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else out += (char)c;
            break;
        }
    }
    return out;
}

static std::string token_text(const llama_vocab *vocab, llama_token tok) {
    std::vector<char> buf(256);
    int n = llama_token_to_piece(vocab, tok, buf.data(), (int)buf.size(), 0, true);
    if (n < 0) {
        buf.resize((size_t)(-n) + 8);
        n = llama_token_to_piece(vocab, tok, buf.data(), (int)buf.size(), 0, true);
    }
    if (n < 0) throw std::runtime_error("failed to convert token to piece");
    return std::string(buf.data(), (size_t)n);
}

static void print_json_bytes(FILE *fp, const std::string &s) {
    std::fputc('[', fp);
    for (size_t i = 0; i < s.size(); i++) {
        if (i) std::fputs(", ", fp);
        std::fprintf(fp, "%u", (unsigned char)s[i]);
    }
    std::fputc(']', fp);
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab *vocab, const std::string &prompt) {
    int n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), nullptr, 0, false, true);
    if (n >= 0) throw std::runtime_error("unexpected non-negative token count probe");
    std::vector<llama_token> tokens((size_t)(-n));
    int got = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), tokens.data(), (int)tokens.size(), false, true);
    if (got < 0) throw std::runtime_error("failed to tokenize prompt");
    tokens.resize((size_t)got);
    return tokens;
}

static void batch_set(struct llama_batch &batch, const std::vector<llama_token> &tokens, int pos0) {
    batch.n_tokens = (int32_t)tokens.size();
    static llama_seq_id seq0 = 0;
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        batch.token[i] = tokens[(size_t)i];
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq0;
        batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
    }
}

static std::vector<token_prob> topk_from_logits(const float *logits, int n_vocab, int k) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_vocab; i++) max_logit = std::max(max_logit, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) sum += std::exp((double)logits[i] - (double)max_logit);
    double logsum = std::log(sum) + (double)max_logit;

    std::vector<token_prob> all;
    all.reserve((size_t)n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        all.push_back({i, logits[i], (float)((double)logits[i] - logsum)});
    }
    std::partial_sort(all.begin(), all.begin() + std::min(k, n_vocab), all.end(),
                      [](const token_prob &a, const token_prob &b) { return a.logit > b.logit; });
    if ((int)all.size() > k) all.resize((size_t)k);
    return all;
}

static std::string render_legacy_chat_prompt(const std::string &user_prompt) {
    return std::string("<|endoftext|><|im_start|>user\n") + user_prompt +
           "<|im_end|>\n<|im_start|>assistant\n</think>";
}

static std::string render_hf_chat_prompt(const std::string &user_prompt) {
    return std::string("<|im_start|>user\n") + user_prompt +
           "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

int main(int argc, char **argv) {
    const char *model_path = nullptr;
    const char *prompt_file = nullptr;
    const char *out_path = nullptr;
    int ctx = 4096;
    int n_predict = 4;
    int top_k = 20;
    int threads = 1;
    bool hf_template = false;

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_file = argv[++i];
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--n-predict") && i + 1 < argc) n_predict = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--hf-template")) hf_template = true;
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            std::puts("usage: llama_qwen_logprobs_capture --model FILE --prompt-file FILE --out FILE [--ctx N] [--n-predict N] [--top-k N] [--threads N] [--hf-template]");
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }

    if (!model_path || !prompt_file || !out_path) {
        std::fprintf(stderr, "missing required arguments\n");
        return 2;
    }

    try {
        llama_backend_init();

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_mmap = true;
        mp.use_mlock = false;
        /* Repacked (q4_K_8x8) weights live in anonymous RAM instead of the
         * mmap; a 20GB model then OOMs the 14GB box. Stay on the mmap. */
        mp.use_extra_bufts = false;
        mp.check_tensors = false;
        llama_model *model = llama_model_load_from_file(model_path, mp);
        if (!model) throw std::runtime_error("failed to load model");

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t)ctx;
        cp.n_batch = (uint32_t)ctx;
        cp.n_ubatch = (uint32_t)std::min(ctx, 2048);
        cp.n_threads = (uint32_t)threads;
        cp.n_threads_batch = (uint32_t)threads;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.no_perf = true;
        llama_context *lctx = llama_init_from_model(model, cp);
        if (!lctx) throw std::runtime_error("failed to create context");

        const llama_vocab *vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        std::string user_prompt = read_file(prompt_file);
        std::string rendered = hf_template ? render_hf_chat_prompt(user_prompt)
                                           : render_legacy_chat_prompt(user_prompt);
        std::vector<llama_token> prompt_tokens = tokenize_prompt(vocab, rendered);

        llama_batch batch = llama_batch_init(ctx, 0, 1);
        batch_set(batch, prompt_tokens, 0);
        if (llama_decode(lctx, batch) != 0) throw std::runtime_error("prompt decode failed");

        FILE *fp = std::fopen(out_path, "wb");
        if (!fp) throw std::runtime_error("failed to open output file");

        std::fprintf(fp, "{\n");
        std::fprintf(fp, "  \"prompt_tokens\": %zu,\n", prompt_tokens.size());
        std::fprintf(fp, "  \"ctx\": %d,\n", ctx);
        std::fprintf(fp, "  \"top_k\": %d,\n", top_k);
        std::fprintf(fp, "  \"template\": \"%s\",\n", hf_template ? "hf-text-only" : "legacy-q36-nothink");
        std::fprintf(fp, "  \"steps\": [\n");

        int pos = (int)prompt_tokens.size();
        for (int step = 0; step < n_predict; step++) {
            const float *logits = llama_get_logits_ith(lctx, batch.n_tokens - 1);
            if (!logits) throw std::runtime_error("missing logits");
            std::vector<token_prob> top = topk_from_logits(logits, n_vocab, top_k);
            llama_token selected = top.empty() ? LLAMA_TOKEN_NULL : top[0].id;
            std::string selected_text = token_text(vocab, selected);

            if (step) std::fprintf(fp, ",\n");
            std::fprintf(fp, "    {\n");
            std::fprintf(fp, "      \"step\": %d,\n", step);
            std::fprintf(fp, "      \"selected\": {\n");
            std::fprintf(fp, "        \"id\": %d,\n", (int)selected);
            std::fprintf(fp, "        \"text\": \"%s\",\n", json_escape(selected_text).c_str());
            std::fprintf(fp, "        \"bytes\": ");
            print_json_bytes(fp, selected_text);
            std::fprintf(fp, "\n      },\n");
            std::fprintf(fp, "      \"top_logprobs\": [\n");
            for (size_t i = 0; i < top.size(); i++) {
                std::string text = token_text(vocab, top[i].id);
                if (i) std::fprintf(fp, ",\n");
                std::fprintf(fp, "        {\n");
                std::fprintf(fp, "          \"token\": {\n");
                std::fprintf(fp, "            \"id\": %d,\n", (int)top[i].id);
                std::fprintf(fp, "            \"text\": \"%s\",\n", json_escape(text).c_str());
                std::fprintf(fp, "            \"bytes\": ");
                print_json_bytes(fp, text);
                std::fprintf(fp, "\n          },\n");
                std::fprintf(fp, "          \"logit\": %.9g,\n", top[i].logit);
                std::fprintf(fp, "          \"logprob\": %.9g\n", top[i].logprob);
                std::fprintf(fp, "        }");
            }
            std::fprintf(fp, "\n      ]\n");
            std::fprintf(fp, "    }");

            if (selected == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, selected)) break;
            std::vector<llama_token> one = {selected};
            batch_set(batch, one, pos++);
            if (llama_decode(lctx, batch) != 0) throw std::runtime_error("decode step failed");
        }
        std::fprintf(fp, "\n  ]\n}\n");
        std::fclose(fp);

        llama_batch_free(batch);
        llama_free(lctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "llama_qwen_logprobs_capture: %s\n", e.what());
        return 1;
    }
}
