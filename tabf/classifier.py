"""Rule-based task-type and complexity classifier."""

from typing import Tuple
from .extractor import Signals


TASK_TYPES = [
    "code_generation",
    "bug_fix",
    "refactoring",
    "documentation",
    "multi_step_plan",
    "data_analysis",
    "research_query",
    "tool_use_chain",
]

COMPLEXITY_LEVELS = ["low", "medium", "high", "expert"]


_BUG_HINTS = ("bug", "error", "crash", " fix", "broken", "fail", "502",
              "intermittent")
_REFAC_HINTS = ("refactor", "restructure", "rewrite", "clean up", "tidy",
                "extract", "split the")
_DOC_HINTS = ("document the", "documentation", "write a readme",
              "readme", "tutorial", "operator guide", "user guide",
              "explain how", "describe how")
_ANALYSIS_HINTS = ("analyse", "analyze", "analysis", "plot", "histogram",
                   "statistics", "correlation", "dataset", "load",
                   "summary", "summarise", "summarize", "produce a summary",
                   "trends", "anomalies", "retention", "report")
_RESEARCH_HINTS = ("survey", "literature", "what is", "what are", "compare",
                   "research", "find papers", "find recent")


def _contains_any(text_low: str, hints) -> bool:
    return any(h in text_low for h in hints)


def classify(task: str, signals: Signals) -> Tuple[str, str]:
    """Return (task_type, complexity_level)."""
    low = (task or "").lower()

    # Task-type rules, priority-ordered. Data analysis is checked before
    # tool_use_chain because some analytical tasks legitimately invoke tools
    # (e.g. "join with the deployment log"), and before documentation so a
    # task like "load X.csv and produce a summary" lands in data_analysis,
    # not docs.
    if signals.has_code_keywords and _contains_any(low, _BUG_HINTS):
        ktype = "bug_fix"
    elif signals.has_code_keywords and _contains_any(low, _REFAC_HINTS):
        ktype = "refactoring"
    elif signals.has_file_refs and _contains_any(low, _ANALYSIS_HINTS):
        ktype = "data_analysis"
    elif _contains_any(low, _DOC_HINTS):
        ktype = "documentation"
    elif signals.has_tool_keywords and signals.has_multi_step:
        ktype = "tool_use_chain"
    elif (signals.has_code_keywords and signals.has_tool_keywords
          and not signals.has_multi_step):
        # "Build a client ... with GET, POST endpoints" is code generation.
        ktype = "code_generation"
    elif signals.has_tool_keywords:
        ktype = "tool_use_chain"
    elif _contains_any(low, _RESEARCH_HINTS) or signals.question_count >= 2:
        ktype = "research_query"
    elif signals.has_multi_step:
        ktype = "multi_step_plan"
    elif signals.has_code_keywords:
        ktype = "code_generation"
    else:
        ktype = "research_query"

    # Complexity score, additive across signals. Thresholds calibrated on
    # the 240-task calibration corpus described in Section 5.
    score = 0
    wc = signals.word_count
    if wc <= 10:
        score += 0
    elif wc <= 20:
        score += 1
    elif wc <= 32:
        score += 2
    elif wc <= 50:
        score += 3
    else:
        score += 4

    score += int(signals.has_multi_step)
    score += int(signals.has_tool_keywords)
    score += int(signals.has_file_refs)
    score += int(signals.has_comparison)
    score += int(signals.has_code_keywords)

    if signals.entity_count >= 6:
        score += 3
    elif signals.entity_count >= 3:
        score += 2
    elif signals.entity_count >= 1:
        score += 1

    score += min(signals.question_count, 2)

    if score <= 2:
        cx = "low"
    elif score <= 4:
        cx = "medium"
    elif score <= 7:
        cx = "high"
    else:
        cx = "expert"

    return ktype, cx
