#pragma once

#include <llama.h>
#include <vector>
#include <string>
#include <functional>

class llm_interface {

public:
    struct model_config {
        double min_p;
        double temperature;
        int context_size;
    };

private:
    llama_context* _ctx;
    llama_model* _model;
    const llama_vocab * _vocab;
    llama_sampler* _sampler;
    llama_batch _batch;
    llama_token _currToken;

    std::vector<llama_chat_message> _messages;
    std::vector<char> _formatted_messages;
    std::vector<llama_token> _prompt_token;

    int _prev_len = 0;

    std::string _response = "";

public:
    llm_interface();
    ~llm_interface();
    void set_system_prompt(const std::string& system_prompt);
    bool load_model(const std::string& model_root_path, const std::string& model_name, const model_config& config);
    bool run_prompt(const std::string& prompt, std::string& result, std::function<void(std::string)> token_out = nullptr);

private:
    std::string begin_prepare_prompt(const std::string& prompt);
    void after_prepare_prepare(const std::string& result);
};