#include "rag_client.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace 
{
    inline void trim(std::string &s) 
    {
        while (!s.empty() && (unsigned char)s.front() <= ' ') s.erase(s.begin());
        while (!s.empty() && (unsigned char)s.back()  <= ' ') s.pop_back();
    }

    static std::vector<float> parse_vec_csv(const std::string& csv) 
    {
        std::vector<float> v;
        v.reserve(2048);
        const char* p = csv.c_str();
        const char* end = p + csv.size();
        while (p < end) {
            while (p < end && (*p == ',' || *p == ' ' || *p == '\t')) ++p;
            if (p >= end) break;
            const char* q = p;
            while (q < end && *q != ',') ++q;
            try {
                v.push_back(std::stof(std::string(p, q)));
            } catch (...) {
                v.push_back(0.0f);
            }
            p = (q < end) ? q + 1 : q;
        }
        return v;
    }
}

bool rag_client::parse_index_line(const std::string& line, rag_index_row& row) 
{
    size_t p1 = line.find('\t');
    if (p1 == std::string::npos) return false;
    size_t p2 = line.find('\t', p1 + 1);
    if (p2 == std::string::npos) return false;
    size_t p3 = line.find('\t', p2 + 1);
    if (p3 == std::string::npos) return false;

    std::string id    = line.substr(0, p1);
    std::string vcsv  = line.substr(p1 + 1, p2 - p1 - 1);
    std::string fname = line.substr(p2 + 1, p3 - p2 - 1);
    std::string text  = line.substr(p3 + 1);

    trim(id); trim(vcsv); trim(fname);
    if (id.empty() || vcsv.empty() || fname.empty()) return false;

    rag_index_row r;
    try { r.id = std::stoi(id); } catch (...) { return false; }

    r.vec      = parse_vec_csv(vcsv);
    r.filename = std::move(fname);
    r.text     = std::move(text);

    if (r.vec.empty()) return false;
    row = std::move(r);
    return true;
}

bool rag_client::load_index(const std::string& index_path) {
    items_.clear();
    std::ifstream fin(index_path);
    if (!fin) return false;

    std::string line;
    items_.reserve(512);
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        rag_index_row row;
        if (parse_index_line(line, row)) {
            items_.push_back(std::move(row));
        }
    }
    return !items_.empty();
}

bool rag_client::load_models(const rag_config& cfg) {
    cfg_ = cfg;

    if (!_embed.load_model(cfg.embed_model_root, cfg.embed_model_name, cfg.embed)) {
        std::cerr << "_embed.load_model failed: " << cfg.embed_model_name << "\n";
        return false;
    }

    _embed.create_index("../rag/docs", "../rag/index.tsv");

    if (!_llm.load_model(cfg.llm_model_root, cfg.llm_model_name, cfg.llm)) {
        std::cerr << "_llm.load_model failed: " << cfg.llm_model_name << "\n";
        return false;
    }
    _models_ready = true;

    _llm.set_system_prompt(cfg.system_prompt);
    return true;
}

bool rag_client::embed_question(const std::string& question, std::vector<float>& out_qvec) const {
    out_qvec.clear();
    if (!_models_ready) return false;
    return _embed.embed_query(question, out_qvec) && !out_qvec.empty();
}

std::vector<rag_client::rag_rank_item> rag_client::rank(const std::vector<float>& qvec) const {
    std::vector<rag_rank_item> ranked;

    ranked.reserve(items_.size());
    for (int i = 0; i < (int)items_.size(); ++i) {
        float s = dot(qvec, items_[i].vec);
        if (cfg_.min_score_keep >= 0.0f && s < cfg_.min_score_keep) continue;
        ranked.push_back({s, i});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const rag_rank_item& a, const rag_rank_item& b) {
                  return a.score > b.score;
              });

    return ranked;
}

std::string rag_client::build_context(const std::vector<rag_rank_item>& ranked,
                                     int top_k,
                                     std::size_t char_budget) const {
    std::ostringstream oss;
    std::size_t used = 0;
    int count = 0;

    for (const auto& it : ranked) {
        if (count >= top_k) break;
        const rag_index_row& r = items_[it.row_index];
        std::string one = "- [" + r.filename + "] " + r.text + "\n\n";
        if (used + one.size() > char_budget && count >= 1) break;
        oss << one;
        used += one.size();
        ++count;
    }
    return oss.str();
}

std::string rag_client::ask(const std::string& question,
                           std::optional<int> override_top_k,
                           std::function<void(const std::string&)> on_token) {
    if (!_models_ready) return "[ERROR] models not loaded";
    if (items_.empty())  return "[ERROR] index is empty";

    std::vector<float> qvec;
    if (!embed_question(question, qvec)) {
        return "[ERROR] failed to embed question";
    }

    auto ranked = rank(qvec);
    if (ranked.empty()) {
        return "[WARN] no relevant context found";
    }

    const int K = override_top_k.value_or(cfg_.top_k);
    std::string ctx = build_context(ranked, K, cfg_.context_budget);

    std::ostringstream user_prompt;
    user_prompt
        << cfg_.system_prompt << "\n\n"
        << "บริบท:\n" << ctx << "\n"
        << "คำถาม: " << question << "\n\n"
        << "ข้อกำหนดการตอบ:\n"
        << "- ตอบเป็นภาษาไทยแบบกระชับ ชัดเจน\n";
        //<< "- หากอ้างอิงข้อมูล ให้ใส่รายการไฟล์อ้างอิง (รูปแบบ [filename#pX]) ท้ายคำตอบ\n";

    std::string final_answer;
    if (cfg_.stream_tokens && on_token) {
        bool ok = _llm.run_prompt(user_prompt.str(), final_answer,
                                  [&](std::string tok){ on_token(tok); });
        if (!ok) return "[ERROR] LLM run_prompt failed";
        return final_answer;
    } else {
        std::string buf;
        bool ok = _llm.run_prompt(user_prompt.str(), final_answer,
                                  [&](std::string tok){ buf += tok; });
        if (!ok) return "[ERROR] LLM run_prompt failed";
        return final_answer.empty() ? buf : final_answer;
    }
}
