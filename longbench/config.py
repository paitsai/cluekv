"""LongBench task definitions, metrics, prompt templates, and model configurations.

References:
    Bai et al. "LongBench: A Bilingual, Multitask Benchmark for Long Context
    Understanding." arXiv:2308.14508.

Prompt templates and max-generation lengths are taken from the official
LongBench repository (config/dataset2prompt.json, config/dataset2maxlen.json).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Subtask groupings (mirrors the paper §5.2 / Table 3)
# ---------------------------------------------------------------------------

SUBTASKS = {
    # Single-Document QA
    "narrativeqa":        {"type": "qa", "metric": "f1"},
    "qasper":             {"type": "qa", "metric": "f1"},
    "multifieldqa_en":    {"type": "qa", "metric": "f1"},

    # Multi-Document QA
    "hotpotqa":           {"type": "qa", "metric": "f1"},
    "2wikimqa":           {"type": "qa", "metric": "f1"},
    "musique":            {"type": "qa", "metric": "f1"},

    # Summarization
    "gov_report":         {"type": "summarization", "metric": "rouge-l"},
    "qmsum":              {"type": "summarization", "metric": "rouge-l"},
    "multi_news":         {"type": "summarization", "metric": "rouge-l"},

    # Few-shot
    "trec":               {"type": "fewshot", "metric": "accuracy"},
    "triviaqa":           {"type": "fewshot", "metric": "f1"},
    "samsum":             {"type": "fewshot", "metric": "rouge-l"},

    # Code
    "lcc":                {"type": "code", "metric": "edit_sim"},
    "repobench-p":        {"type": "code", "metric": "edit_sim"},

    # Synthetic
    "passage_retrieval_en": {"type": "synthetic", "metric": "retrieval"},
    "passage_retrieval_zh": {"type": "synthetic", "metric": "retrieval_zh"},
    "multifieldqa_zh":      {"type": "qa", "metric": "f1"},
}

# Short readable names for table headers (mirrors paper Table 3)
SUBTASK_ALIASES = {
    "narrativeqa":        "QA",
    "qasper":             "Qas",
    "multifieldqa_en":    "MF-en",
    "hotpotqa":           "Hot",
    "2wikimqa":           "2Wi",
    "musique":            "Mus",
    "gov_report":         "Gov",
    "qmsum":              "QMS",
    "multi_news":         "MN",
    "trec":               "Trec",
    "triviaqa":           "Tri",
    "samsum":             "Sam",
    "lcc":                "Lcc",
    "repobench-p":        "RB-P",
    "passage_retrieval_en": "PR-en",
    "passage_retrieval_zh": "PR-zh",
    "multifieldqa_zh":    "MF-zh",
}

# ---------------------------------------------------------------------------
# Prompt templates — matches official LongBench config/dataset2prompt.json
# Placeholders: {context} = passage/document, {input} = question/query
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script,"
        " and a question. Answer the question asconcisely as you can, using a"
        " single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can,"
        " using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question"
        " as concisely as you can, using a single phrase or sentence if possible."
        " If the question cannot be answered based on the information in the"
        ' article, write "unanswerable". If the question is a yes/no question,'
        ' answer "yes", "no", or "unanswerable". Do not provide any explanation.'
        "\n\n"
        "Article: {context}\n\n"
        " Answer the question based on the above article as concisely as you can,"
        " using a single phrase or sentence if possible. If the question cannot be"
        ' answered based on the information in the article, write "unanswerable".'
        ' If the question is a yes/no question, answer "yes", "no", or'
        ' "unanswerable". Do not provide any explanation.\n\n'
        "Question: {input}\n\n"
        "Answer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, only give me"
        " the answer and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "multifieldqa_zh": (
        "阅读以下文字并用中文简短回答：\n\n"
        "{context}\n\n"
        "现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n"
        "问题：{input}\n"
        "回答："
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer"
        " and do not output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer"
        " and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer"
        " and do not output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer"
        " and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer"
        " and do not output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer"
        " and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary"
        " of the report.\n\n"
        "Report:\n"
        "{context}\n\n"
        "Now, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or"
        " instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n"
        "{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or"
        " more sentences.\n\n"
        "Query: {input}\n"
        "Answer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all"
        " news. \n\n"
        "News:\n"
        "{context}\n\n"
        "Now, write a one-page summary of all the news.\n\n"
        "Summary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples"
        " of questions.\n\n"
        "{context}\n"
        "{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer"
        " and do not output any other words. The following are some examples.\n\n"
        "{context}\n\n"
        "{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are"
        " some examples.\n\n"
        "{context}\n\n"
        "{input}"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please"
        " determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n"
        "{input}\n\n"
        'Please enter the number of the paragraph that the abstract is from.'
        ' The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\n'
        "The answer is: "
    ),
    "passage_retrieval_zh": (
        "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n"
        "{context}\n\n"
        "下面是一个摘要\n\n"
        "{input}\n\n"
        '请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n'
        "答案是："
    ),
    "lcc": (
        "Please complete the code given below. \n"
        "{context}"
        "Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below. \n"
        "{context}{input}"
        "Next line of code:\n"
    ),
}

# ---------------------------------------------------------------------------
# Max generation length per dataset — matches official config/dataset2maxlen.json
# ---------------------------------------------------------------------------

DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

# ---------------------------------------------------------------------------
# Model max context length (input truncation threshold)
# ---------------------------------------------------------------------------

MODEL2MAXLEN = {
    "meta-llama/Llama-3.1-8B-Instruct": 3500,
    "Qwen/Qwen3-8B": 31500,
    "Qwen/Qwen3-1.7B": 31500,
    "mistralai/Mistral-7B-Instruct-v0.3": 31500,
}

# ---------------------------------------------------------------------------
# Generation config per model
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "max_length": 3500,
        "gen_kwargs": dict(do_sample=False, temperature=1.0),
        "chat_template": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n\n"
            "{input}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    "Qwen/Qwen3-8B": {
        "max_length": 31500,
        "gen_kwargs": dict(do_sample=False, temperature=1.0),
        "chat_template": (
            "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        ),
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "max_length": 31500,
        "gen_kwargs": dict(do_sample=False, temperature=1.0),
        "chat_template": (
            "<s>[INST] {input} [/INST]"
        ),
    },
}

# ---------------------------------------------------------------------------
# ClueKV parameters for LongBench (from paper Appendix A)
# ---------------------------------------------------------------------------

CLUEKV_LONGBENCH_PARAMS = dict(
    kappa=128,
    tau=1.0,
    chunk_size=256,
    sink_tokens=8,
    obs_window_size=32,
    cache_budget_ratio=0.25,
    attn_compute_chunk=4,
    pool_mode="avg",
    smooth_kernel=9,
)

# Cache budgets as token counts (paper Table 3)
CACHE_BUDGETS = [512, 1024, 2048]  # tokens
