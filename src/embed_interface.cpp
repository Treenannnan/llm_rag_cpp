#include "embed_interface.h"
#include <format>
#include <cstring>
#include <cmath>
#include <iostream>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>

embed_interface::embed_interface() {
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            std::fprintf(stderr, "%s", text);
        }
    }, nullptr);

    llama_backend_init();
}

embed_interface::~embed_interface() {
    if (_ctx)   llama_free(_ctx);
    if (_model) llama_model_free(_model);
    llama_backend_free();
}

bool embed_interface::load_model(const std::string& model_root_path,
                                 const std::string& model_name,
                                 const model_config& cfg) {
    _cfg = cfg;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.n_gpu_layers;

    const std::string model_path = std::format("{}/{}", model_root_path, model_name);
    _model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!_model) {
        std::fprintf(stderr, "[embed] failed to load model: %s\n", model_path.c_str());
        return false;
    }

    _vocab = llama_model_get_vocab(_model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = cfg.context_size;
    cparams.n_batch    = cfg.n_batch;
    cparams.embeddings = true;

    cparams.pooling_type = cfg.use_mean_pool ? LLAMA_POOLING_TYPE_MEAN : LLAMA_POOLING_TYPE_NONE;

    _ctx = llama_init_from_model(_model, cparams);
    if (!_ctx) {
        std::fprintf(stderr, "[embed] llama_init_from_model() returned null\n");
        return false;
    }

    _n_embd = llama_model_n_embd(_model);
    if (_n_embd <= 0) {
        std::fprintf(stderr, "[embed] invalid n_embd\n");
        return false;
    }

    std::fprintf(stderr, "[embed] ctx: n_ctx=%d n_batch=%d pooling=%d (MEAN=2) embd_dim=%d\n",
                 llama_n_ctx(_ctx), cparams.n_batch, (int)llama_pooling_type(_ctx), _n_embd);

    return true;
}

bool embed_interface::embed_query(const std::string& text, std::vector<float>& out) const 
{
    return encode_once(_cfg.query_prefix + text, out);
}

bool embed_interface::embed_passage(const std::string& text, std::vector<float>& out) const 
{
    return encode_once(_cfg.passage_prefix + text, out);
}

bool embed_interface::embed_batch(const std::vector<std::string>& texts,
                                  std::vector<std::vector<float>>& out) {
    out.clear();
    out.reserve(texts.size());
    for (const auto& t : texts) {
        std::vector<float> v;
        if (!encode_once(t, v)) return false;
        out.push_back(std::move(v));
    }
    return true;
}

bool embed_interface::encode_once(const std::string& text, std::vector<float>& out_emb) const {
    out_emb.clear();

    llama_memory_clear(llama_get_memory(_ctx), true);

    std::string in = _cfg.query_prefix.empty() ? text : (_cfg.query_prefix + text);

    std::vector<llama_token> toks = embd_tokenize(_vocab, in, true, true);
    if (toks.empty()) {
        out_emb.assign(_n_embd, 0.0f);
        return true;
    }

    if (_cfg.add_bos && llama_vocab_get_add_bos(_vocab)) {
        const llama_token bos_id = llama_vocab_bos(_vocab);
        if (toks.empty() || toks.front() != bos_id) {
            toks.insert(toks.begin(), bos_id);
        }
    }

    if ((int)toks.size() > llama_n_ctx(_ctx)) {
        std::fprintf(stderr, "[embed] too many tokens: %d > n_ctx %d\n", (int)toks.size(), llama_n_ctx(_ctx));
        return false;
    }

    const int n_ctx = llama_n_ctx(_ctx);
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    batch.n_tokens = 0;
    const std::vector<llama_seq_id> seq_ids{0};
    for (int i = 0; i < (int)toks.size(); ++i) 
    {
        batch.token   [batch.n_tokens] = toks[i];
        batch.pos     [batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = seq_ids.size();
        for (size_t i = 0; i < seq_ids.size(); ++i) {
            batch.seq_id[batch.n_tokens][i] = seq_ids[i];
        }
        batch.logits  [batch.n_tokens] = true;

        batch.n_tokens++;
    }

    if (llama_decode(_ctx, batch) < 0) {
        std::fprintf(stderr, "[embed] llama_decode failed\n");
        llama_batch_free(batch);
        return false;
    }

    const enum llama_pooling_type pooling = llama_pooling_type(_ctx);
    const float* embd_ptr = nullptr;

    if (pooling == LLAMA_POOLING_TYPE_NONE) 
    {
        embd_ptr = llama_get_embeddings_ith(_ctx, (int)toks.size() - 1);
        if (!embd_ptr) {
            std::fprintf(stderr, "[embed] get_embeddings_ith returned null\n");
            llama_batch_free(batch);
            return false;
        }
    } 
    else 
    {
        embd_ptr = llama_get_embeddings_seq(_ctx, /*seq_id=*/0);
        if (!embd_ptr) 
        {
            std::fprintf(stderr, "[embed] get_embeddings_seq returned null\n");
            llama_batch_free(batch);
            return false;
        }
    }

    out_emb.resize(_n_embd);
    const int embd_norm = _cfg.normalize_l2 ? 2 : 0;
    embd_normalize(embd_ptr, out_emb.data(), _n_embd, embd_norm);

    double sumsq = 0.0, sumabs = 0.0;
    for (int i = 0; i < _n_embd; ++i) 
    { 
        sumsq += (double)out_emb[i]*out_emb[i]; sumabs += std::fabs(out_emb[i]); 
    }

    llama_batch_free(batch);
    return true;
}

bool embed_interface::create_index(const std::string& docs_path, const std::string index_output_path)
{
    namespace fs = std::filesystem;

    auto clean_spaces = [](std::string s) -> std::string {
        s = std::regex_replace(s, std::regex("\\s+"), " ");
        auto l = s.find_first_not_of(' ');
        auto r = s.find_last_not_of(' ');
        if (l == std::string::npos) return "";
        return s.substr(l, r - l + 1);
    };

    auto chunk_words = [&clean_spaces](const std::string& text, int max_words, int overlap) -> std::vector<std::string> {
        std::vector<std::string> out;
        std::vector<std::string> words;
        words.reserve(text.size() / 5);
        {
            std::istringstream iss(text);
            std::string w;
            while (iss >> w) words.push_back(std::move(w));
        }
        if (words.empty()) return out;

        const int step = std::max(1, max_words - overlap);
        for (int i = 0; i < (int)words.size(); i += step) {
            const int j = std::min<int>((int)words.size(), i + max_words);
            std::ostringstream oss;
            for (int k = i; k < j; ++k) {
                if (k > i) oss << ' ';
                oss << words[k];
            }
            auto c = clean_spaces(oss.str());
            if (!c.empty()) out.push_back(std::move(c));
        }
        return out;
    };

    const int MAX_WORDS = 1000;
    const int OVERLAP   = 80;

    if (!_ctx || !_model) {
        std::fprintf(stderr, "[embed] create_index: model/context not initialized. Call load_model() first.\n");
        return false;
    }

    std::ofstream fout(index_output_path, std::ios::binary);
    if (!fout) {
        std::fprintf(stderr, "[embed] create_index: cannot open output: %s\n", index_output_path.c_str());
        return false;
    }

    size_t chunk_id = 0;
    size_t file_cnt = 0;
    size_t chunk_cnt = 0;

    try 
    {
        for (auto& entry : fs::recursive_directory_iterator(docs_path)) 
        {
            if (!entry.is_regular_file()) continue;

            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".txt" && ext != ".md") continue;

            std::ifstream fin(entry.path());
            if (!fin) continue;

            std::string all((std::istreambuf_iterator<char>(fin)), {});
            all = clean_spaces(all);
            if (all.empty()) continue;

            ++file_cnt;
            auto chunks = chunk_words(all, MAX_WORDS, OVERLAP);

            for (auto& ch : chunks) {
                std::vector<float> emb;
                if (!this->embed_passage(ch, emb)) {
                    std::fprintf(stderr, "[embed] create_index: embed() failed for chunk in %s\n",
                                 entry.path().string().c_str());
                    return false;
                }

                std::ostringstream vcsv;
                vcsv.setf(std::ios::fixed);
                vcsv.precision(7);
                for (size_t i = 0; i < emb.size(); ++i) {
                    if (i) vcsv << ',';
                    vcsv << emb[i];
                }

                fout << chunk_id++ 
                << '\t' << vcsv.str() 
                << '\t' << entry.path().filename().string()
                << '\t' << ch 
                << '\n';
                ++chunk_cnt;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[embed] create_index: exception: %s\n", e.what());
        return false;
    }

    fout.close();
    std::fprintf(stderr, "[embed] create_index: wrote %zu chunks from %zu files -> %s\n",
                 chunk_cnt, file_cnt, index_output_path.c_str());
    return true;
}

std::vector<llama_token> embed_interface::embd_tokenize(const struct llama_vocab* vocab, const std::string& text, bool add_special, bool parse_special) const
{
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) 
    {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) 
    {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } 
    else 
    {
        result.resize(n_tokens);
    }
    return result;
}

void embed_interface::embd_normalize(const float * inp, float * out, int n, int embd_norm) const
{
    double sum = 0.0;

    switch (embd_norm) 
    {
        case -1:
            sum = 1.0;
            break;
        case 0:
            for (int i = 0; i < n; i++) 
            {
                if (sum < std::abs(inp[i])) 
                {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0;
            break;
        case 2:
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default:
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) 
    {
        out[i] = inp[i] * norm;
    }
}