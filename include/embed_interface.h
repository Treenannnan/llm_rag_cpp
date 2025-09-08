#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include "llama.h"

class embed_interface {
public:
    struct model_config {
        int   context_size   = 4096;
        int   n_batch        = 2048;
        int   n_gpu_layers   = 99;
        bool  normalize_l2   = true;

        bool  use_mean_pool  = true;
        bool  add_bos        = true;
        bool  add_special    = false;

        std::string query_prefix = "";
        std::string passage_prefix = "";
        
    };

    embed_interface();
    ~embed_interface();

    bool load_model(const std::string& model_root_path,
                    const std::string& model_name,
                    const model_config& cfg);

    bool embed_query(const std::string& text, std::vector<float>& out) const;
    bool embed_passage(const std::string& text, std::vector<float>& out) const;

    bool embed_batch(const std::vector<std::string>& texts,
                     std::vector<std::vector<float>>& out);

    int  dim() const { return _n_embd; }

    bool create_index(const std::string& docs_path, const std::string index_output_path);

private:
    bool encode_once(const std::string& text, std::vector<float>& out_emb) const; 
    void embd_normalize(const float * inp, float * out, int n, int embd_norm) const;
    std::vector<llama_token> embd_tokenize(const struct llama_vocab* vocab, const std::string& text, bool add_special, bool parse_special) const;

private:
    llama_model*        _model   = nullptr;
    llama_context*      _ctx     = nullptr;
    const llama_vocab*  _vocab   = nullptr;
    model_config        _cfg{};
    int                 _n_embd  = 0;
};
