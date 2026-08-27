"""LongBench evaluation metrics — matches the official LongBench/metrics.py.

Key: the official implementation uses the ``rouge`` library (not rouge-score),
``fuzzywuzzy`` for code similarity, and ``jieba`` for Chinese tokenization.
"""

from __future__ import annotations

import re
import string
from collections import Counter

# Optional dependencies (match official LongBench)
try:
    from rouge import Rouge
    _ROUGE = Rouge()
    _HAS_ROUGE = True
except ImportError:
    _ROUGE = None
    _HAS_ROUGE = False

try:
    from fuzzywuzzy import fuzz
    _HAS_FUZZ = True
except ImportError:
    fuzz = None  # type: ignore
    _HAS_FUZZ = False

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    jieba = None  # type: ignore
    _HAS_JIEBA = False


# ===========================================================================
# Normalisation — matches official normalize_answer / normalize_zh_answer
# ===========================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove articles, punctuation, and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Lower text and remove Chinese + English punctuation and extra whitespace."""

    def white_space_fix(text: str) -> str:
        return "".join(text.split())

    def remove_punc(text: str) -> str:
        cn_punctuation = (
            "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
            "｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—"
            "'" + "'" + '"' + '"' + "‛" + '"' + '"' + "„‟…‧﹏."
        )
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


# ===========================================================================
# Token-level F1 (used by QA tasks)
# ===========================================================================

def _f1_score(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """English QA F1 with article/punctuation removal (official)."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return _f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Chinese QA F1 with jieba tokenization (official)."""
    if not _HAS_JIEBA:
        # Fallback to character-level
        prediction_tokens = list(prediction)
        ground_truth_tokens = list(ground_truth)
    else:
        prediction_tokens = list(jieba.cut(prediction, cut_all=False))
        ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(t) for t in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(t) for t in ground_truth_tokens]
    prediction_tokens = [t for t in prediction_tokens if len(t) > 0]
    ground_truth_tokens = [t for t in ground_truth_tokens if len(t) > 0]
    return _f1_score(prediction_tokens, ground_truth_tokens)


# ===========================================================================
# ROUGE-L — uses the official ``rouge`` library
# ===========================================================================

def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """ROUGE-L F1 using the ``rouge`` library (official)."""
    if not _HAS_ROUGE:
        # Pure-Python fallback
        return _rouge_l_fallback(prediction, ground_truth)
    try:
        scores = _ROUGE.get_scores([prediction], [ground_truth], avg=True)
        return scores["rouge-l"]["f"]
    except Exception:
        return 0.0


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Chinese ROUGE-L with jieba segmentation (official)."""
    if _HAS_JIEBA:
        prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
        ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def _rouge_l_fallback(prediction: str, ground_truth: str) -> float:
    """Pure-Python ROUGE-L F1 (used when ``rouge`` library is unavailable)."""
    pred_tokens = prediction.lower().strip().split()
    gt_tokens = ground_truth.lower().strip().split()
    if not pred_tokens or not gt_tokens:
        return 0.0

    m, n = len(pred_tokens), len(gt_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs_len = prev[n]
    precision = lcs_len / max(len(pred_tokens), 1)
    recall = lcs_len / max(len(gt_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ===========================================================================
# Classification score — official weighted class-name matching
# ===========================================================================

def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Official weighted classification score using all_classes."""
    all_classes = kwargs.get("all_classes", [])
    if not all_classes:
        # Fallback: exact match
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    em_match_list: list[str] = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


# ===========================================================================
# Code similarity — official fuzzywuzzy-based
# ===========================================================================

def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Official code similarity: extract first non-comment line, then fuzzy ratio."""
    all_lines = prediction.lstrip('\n').split('\n')
    code_line = ""
    for line in all_lines:
        if '`' not in line and '#' not in line and '//' not in line:
            code_line = line
            break
    if not code_line:
        code_line = prediction

    if _HAS_FUZZ:
        return fuzz.ratio(code_line, ground_truth) / 100.0

    # Fallback: character-level edit similarity
    return _edit_sim_fallback(code_line, ground_truth)


def _edit_sim_fallback(prediction: str, ground_truth: str) -> float:
    """Character-level edit similarity (fallback when fuzzywuzzy unavailable)."""
    m, n = len(prediction), len(ground_truth)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if prediction[i - 1] == ground_truth[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return 1.0 - prev[n] / max(m, n)


# ===========================================================================
# Retrieval scores
# ===========================================================================

def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """English retrieval: extract paragraph number and compare."""
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0] if matches else ground_truth
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return right_num / max(len(numbers), 1) if numbers else 0.0


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Chinese retrieval: extract paragraph number and compare."""
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0] if matches else ground_truth
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return right_num / max(len(numbers), 1) if numbers else 0.0


# ===========================================================================
# Exact-match accuracy
# ===========================================================================

def compute_accuracy(prediction: str, answers: list[str]) -> float:
    """1.0 if prediction matches any ground-truth answer, else 0.0."""
    pred_norm = normalize_answer(prediction)
    for ans in answers:
        if pred_norm == normalize_answer(ans):
            return 1.0
    return 0.0


# ===========================================================================
# Metric dispatch — matches official dataset2metric
# ===========================================================================

def compute_f1_max(prediction: str, answers: list[str]) -> float:
    """Max F1 over multiple ground-truth answers."""
    return max(qa_f1_score(prediction, ans) for ans in answers)


def compute_f1_zh_max(prediction: str, answers: list[str]) -> float:
    return max(qa_f1_zh_score(prediction, ans) for ans in answers)


def compute_rouge_l_max(prediction: str, answers: list[str]) -> float:
    """Max ROUGE-L over multiple ground-truths."""
    return max(rouge_score(prediction, ans) for ans in answers)


def compute_rouge_zh_max(prediction: str, answers: list[str]) -> float:
    return max(rouge_zh_score(prediction, ans) for ans in answers)


def compute_edit_sim_max(prediction: str, answers: list[str]) -> float:
    return max(code_sim_score(prediction, ans) for ans in answers)


def compute_classification_score(prediction: str, answers: list[str], **kwargs) -> float:
    """Classification score using official weighted matching."""
    return max(classification_score(prediction, ans, **kwargs) for ans in answers)


def compute_retrieval_score(prediction: str, answers: list[str]) -> float:
    return max(retrieval_score(prediction, ans) for ans in answers)


# Top-level dispatch: metric_name → function(prediction, answers)
METRIC_FUNCTIONS = {
    "f1":          compute_f1_max,
    "accuracy":    compute_classification_score,
    "rouge-l":     compute_rouge_l_max,
    "edit_sim":    compute_edit_sim_max,
    "retrieval":   compute_retrieval_score,
    "retrieval_zh": lambda p, a, **kw: max(retrieval_zh_score(p, ans) for ans in a),
}


def compute_metric(prediction: str, answers: list[str], metric_name: str,
                   all_classes: list[str] | None = None) -> float:
    """Dispatch to the appropriate metric function."""
    fn = METRIC_FUNCTIONS.get(metric_name)
    if fn is None:
        raise ValueError(f"Unknown metric: {metric_name}")
    kwargs = {}
    if all_classes is not None:
        kwargs["all_classes"] = all_classes
    return fn(prediction, answers, **kwargs)
