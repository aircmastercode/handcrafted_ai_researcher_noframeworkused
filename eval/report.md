# Deep Research Agent — Evaluation Report

_Dataset size: **2** items_

## Overall

| Metric | Value |
|--------|-------|
| Citation coverage (auto)   | 100% |
| Citation validity (auto)   | 100% |
| Source diversity (auto)    | 1.00 |
| Faithfulness (LLM judge)   | — |
| Relevance (LLM judge)      | — |
| Latency mean (ms)          | 11046 |
| Latency p95 (ms)           | 11900 |

## Per category

| Category | n | Coverage | Validity | Diversity | Faithfulness | Relevance | Category pass | Latency p95 |
|---|---|---|---|---|---|---|---|---|
| factual | 2 | 100% | 100% | 1.00 | — | — | — | 11900 |

## Per item

### `fact-01` — factual
**Question:** Who founded Sarvam AI and when was the company founded?

- coverage: 100% · validity: 100% · diversity: 1.00 · faithfulness: — · relevance: — · latency: 10192 ms
- cited domains: en.wikipedia.org, tracxn.com

**Answer:**

Sarvam AI was founded by Vivek Raghavan and Pratyush Kumar [Sarvam AI - Wikipedia — en.wikipedia.org](https://en.wikipedia.org/wiki/Sarvam_AI)[Sarvam - 2026 Company Profile & Team - Tracxn — tracxn.com](https://tracxn.com/d/companies/sarvam/__pdMzZ7Rkxe_acM5ctqBwOaZ9aoqOdLTKSvAsHq-7kFw). The company was founded in 2023 [Sarvam AI - Wikipedia — en.wikipedia.org](https://en.wikipedia.org/wiki/Sarvam_AI)[Sarvam - 2026 Company Profile & Team - Tracxn — tracxn.com](https://tracxn.com/d/companies/sarvam/__pdMzZ7Rkxe_acM5ctqBwOaZ9aoqOdLTKSvAsHq-7kFw). Specifically, according to [Sarvam AI - Wikipedia — en.wikipedia.org](https://en.wikipedia.org/wiki/Sarvam_AI), Sarvam AI was founded in August 2023.


---

**Sources**
1. [Sarvam AI - Wikipedia — en.wikipedia.org](https://en.wikipedia.org/wiki/Sarvam_AI)
2. [Sarvam - 2026 Company Profile & Team - Tracxn — tracxn.com](https://tracxn.com/d/companies/sarvam/__pdMzZ7Rkxe_acM5ctqBwOaZ9aoqOdLTKSvAsHq-7kFw)

---

### `fact-02` — factual
**Question:** What is the context-window length advertised for Google's Gemini 2.5 Flash model?

- coverage: 100% · validity: 100% · diversity: 1.00 · faithfulness: — · relevance: — · latency: 11900 ms
- cited domains: cloud.google.com, developers.googleblog.com, docs.cloud.google.com, forum.cursor.com

**Answer:**

The context-window length advertised for Google's Gemini 2.5 Flash model is 1M in MAX mode and 128K in non-MAX mode [Gemini 2.5 flash MAX mode has conflicting docs for context window size — forum.cursor.com](https://forum.cursor.com/t/gemini-2-5-flash-max-mode-has-conflicting-docs-for-context-window-size/100367). This information is confirmed by a forum discussion where a representative states that the error is in the documentation, and the model has 1M context window in MAX mode and 128K in non-MAX mode [Gemini 2.5 flash MAX mode has conflicting docs for context window size — forum.cursor.com](https://forum.cursor.com/t/gemini-2-5-flash-max-mode-has-conflicting-docs-for-context-window-size/100367). However, other snippets do not provide explicit information about the context-window length [Continuing to bring you our latest models, with an improved Gemini 2.5 Flash and Flash-Lite release — developers.googleblog.com](https://developers.googleblog.com/en/continuing-to-bring-you-our-latest-models-with-an-improved-gemini-2-5-flash-and-flash-lite-release/)[Gemini 2.5 Flash | Gemini Enterprise Agent Platform | Google Cloud Documentation — docs.cloud.google.com](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-flash)[Gemini 2.5 Updates: Flash/Pro GA, SFT, Flash-Lite on Vertex AI | Google Cloud Blog — cloud.google.com](https://cloud.google.com/blog/products/ai-machine-learning/gemini-2-5-flash-lite-flash-pro-ga-vertex-ai).


---

**Sources**
1. [Gemini 2.5 flash MAX mode has conflicting docs for context window size — forum.cursor.com](https://forum.cursor.com/t/gemini-2-5-flash-max-mode-has-conflicting-docs-for-context-window-size/100367)
2. [Continuing to bring you our latest models, with an improved Gemini 2.5 Flash and Flash-Lite release — developers.googleblog.com](https://developers.googleblog.com/en/continuing-to-bring-you-our-latest-models-with-an-improved-gemini-2-5-flash-and-flash-lite-release/)
3. [Gemini 2.5 Flash | Gemini Enterprise Agent Platform | Google Cloud Documentation — docs.cloud.google.com](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-flash)
4. [Gemini 2.5 Updates: Flash/Pro GA, SFT, Flash-Lite on Vertex AI | Google Cloud Blog — cloud.google.com](https://cloud.google.com/blog/products/ai-machine-learning/gemini-2-5-flash-lite-flash-pro-ga-vertex-ai)

---
