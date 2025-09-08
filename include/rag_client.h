#pragma once
#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <cstddef>
#include <utility>

#include "embed_interface.h"
#include "llm_interface.h"  

class rag_client 
{
public:
    struct rag_index_row 
    {
        int id = -1; 
        std::vector<float> vec;  
        std::string filename;
        std::string text;
    };

    struct rag_rank_item 
    {
        float score = 0.0f; 
        int row_index = -1;
    };

    struct rag_config 
    {
        std::string index_path;

        std::string embed_model_root;
        std::string embed_model_name;
        embed_interface::model_config embed;

        std::string llm_model_root;
        std::string llm_model_name;
        llm_interface::model_config llm;

        int         top_k           = 8;
        std::size_t context_budget  = 3500;
        float       min_score_keep  = -1.0f;

        std::string system_prompt = "คุณคือผู้ช่วย RAG ภาษาไทย ตอบเป็นภาษาไทยเท่านั้น ตอบจากบริบทเท่านั้น ถ้าไม่มีข้อมูลให้บอกว่าไม่ทราบ";
        bool stream_tokens = true;          
    };

private:
    rag_config cfg_{};
    std::vector<rag_index_row>  items_{};
    embed_interface _embed;
    llm_interface _llm;
    bool _models_ready = false;

public:
    rag_client() = default;
    ~rag_client() = default;
    bool load_index(const std::string& index_path);

    bool load_models(const rag_config& cfg);

    void set_config(const rag_config& cfg) { cfg_ = cfg; }

    const rag_config& config() const { return cfg_; }

    std::string ask(const std::string& question,
                    std::optional<int> override_top_k = std::nullopt,
                    std::function<void(const std::string&)> on_token = nullptr);

    bool embed_question(const std::string& question, std::vector<float>& out_qvec) const;

    std::vector<rag_rank_item> rank(const std::vector<float>& qvec) const;

    std::string build_context(const std::vector<rag_rank_item>& ranked,
                              int top_k,
                              std::size_t char_budget) const;

    const std::vector<rag_index_row>& items() const { return items_; } 

private:
    static inline float dot(const std::vector<float>& a, const std::vector<float>& b) 
    {
        const std::size_t n = std::min(a.size(), b.size());
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += (double)a[i] * (double)b[i];
        return (float)s;
    }

    static bool parse_index_line(const std::string& line, rag_index_row& row);
};