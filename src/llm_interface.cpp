#include "llm_interface.h"
#include <format>
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>

llm_interface::llm_interface()
{
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    ggml_backend_load_all();

    _messages.clear();
}
llm_interface::~llm_interface()
{
    llama_sampler_free(_sampler);
    llama_free(_ctx);
    llama_model_free(_model);
    llama_backend_free();
}

void llm_interface::set_system_prompt(const std::string& system_prompt)
{
    _messages.push_back({"system", strdup(system_prompt.c_str())});
}

bool llm_interface::load_model(const std::string& model_root_path, const std::string& model_name, const model_config& config)
{
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    auto model_path = std::format("{}/{}", model_root_path, model_name);
    _model = llama_model_load_from_file(model_path.c_str(), model_params);
    
    if (!_model) 
    {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return false;
    }

    _vocab = llama_model_get_vocab(_model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.context_size;
    ctx_params.n_batch = config.context_size;

    _ctx = llama_init_from_model(_model, ctx_params);

    if (!_ctx) {
        throw std::runtime_error("llama_new_context_with_model() returned null");
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true; 
    _sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(_sampler, llama_sampler_init_min_p(config.min_p, 1));
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(config.temperature));
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    _formatted_messages = std::vector<char>(llama_n_ctx(_ctx));
    
    return true;
}

bool llm_interface::run_prompt(const std::string& prompt, std::string& result, std::function<void(std::string)> token_out) 
{
    result = "";

    llama_memory_clear(llama_get_memory(_ctx), true);

    auto n_prompt = begin_prepare_prompt(prompt);

    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(_ctx), 0) == -1;

    int32_t n_prompt_tokens = -llama_tokenize(_vocab, n_prompt.c_str(), n_prompt.size(), NULL, 0, is_first, true);

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(_vocab, n_prompt.c_str(), n_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) 
    {
        GGML_ABORT("failed to tokenize the prompt\n");
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;

    while (true) 
    {
        int n_ctx = llama_n_ctx(_ctx);
        int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(_ctx), 0) + 1;
        if (n_ctx_used + batch.n_tokens > n_ctx) 
        {
            fprintf(stderr, "context size exceeded\n");
            exit(0);
        }

        int ret = llama_decode(_ctx, batch);
        if (ret != 0) 
        {
            GGML_ABORT("failed to decode, ret = %d\n", ret);
        }

        new_token_id = llama_sampler_sample(_sampler, _ctx, -1);

        if (llama_vocab_is_eog(_vocab, new_token_id)) 
        {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(_vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) 
        {
            GGML_ABORT("failed to convert token to piece\n");
        }
        std::string piece(buf, n);
        if(token_out != nullptr)
            token_out(piece);

        result += piece;

        batch = llama_batch_get_one(&new_token_id, 1);
    }

    after_prepare_prepare(result);

    return true;
}

std::string llm_interface::begin_prepare_prompt(const std::string& prompt)
{
    _messages.push_back({"user", strdup(prompt.c_str())});

    const char * tmpl = llama_model_chat_template(_model, nullptr);

    int new_len = llama_chat_apply_template(tmpl, _messages.data(), _messages.size(), true, _formatted_messages.data(), _formatted_messages.size());
    if (new_len > (int)_formatted_messages.size()) 
    {
        _formatted_messages.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, _messages.data(), _messages.size(), true, _formatted_messages.data(), _formatted_messages.size());
    }
    if (new_len < 0) 
    {
        fprintf(stderr, "failed to apply the chat template\n");
        return "";
    }

    return std::string(_formatted_messages.begin() + _prev_len, _formatted_messages.begin() + new_len);
}

void llm_interface::after_prepare_prepare(const std::string& result)
{
    _messages.push_back({"assistant", strdup(result.c_str())});

    const char * tmpl = llama_model_chat_template(_model, nullptr);

    _prev_len = llama_chat_apply_template(tmpl, _messages.data(), _messages.size(), false, nullptr, 0);
    if (_prev_len < 0) 
    {
        fprintf(stderr, "failed to apply the chat template\n");
        return;
    }
}