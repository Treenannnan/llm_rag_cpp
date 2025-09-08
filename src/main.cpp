#include "rag_client.h"
#include <iostream>

int main() {
    rag_client rag;
    rag_client::rag_config cfg;

    cfg.index_path = "../rag/index.tsv";

    cfg.embed_model_root = "../models";
    cfg.embed_model_name = "bge-m3-q4_k_m.gguf";
    cfg.embed.context_size = 4096;
    cfg.embed.n_batch      = 2048;
    cfg.embed.n_gpu_layers = 99;
    cfg.embed.normalize_l2 = true;
    cfg.embed.use_mean_pool = true;
    cfg.embed.add_bos = true;
    cfg.embed.add_special = false;
    cfg.embed.query_prefix  = "query: ";
    cfg.embed.passage_prefix  = "passage: "; 

    cfg.llm_model_root = "../models";
    cfg.llm_model_name = "openthaigpt1.5-14b-instruct.i1-Q6_K.gguf";
    cfg.llm.context_size = 4096;
    cfg.llm.min_p        = 0.05f;
    cfg.llm.temperature  = 0.3f;

    cfg.top_k = 8;
    cfg.context_budget = 3500;

    rag.set_config(cfg);

    if (!rag.load_models(cfg)) 
    {
        std::cerr << "load_models failed\n";
        return 2;
    }

    if (!rag.load_index(cfg.index_path)) 
    {
        std::cerr << "load_index failed\n";
        return 1;
    }

    while (true)
    {
        std::string input_str;

        printf("\033[32m> \033[0m");
        std::getline(std::cin, input_str);

        rag.ask(input_str, std::nullopt, [](const std::string& tok)
        { 
            std::cout << tok << std::flush; 
        });

        std::cout << std::endl;
    }
    
    std::cout << "\n\n=== DONE ===\n";
    return 0;
}
