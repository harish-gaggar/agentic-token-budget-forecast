"""Lexical signal extraction. No tokenizer, no model, no dependencies."""

import re
from dataclasses import dataclass


MULTI_STEP = {
    "then", "next", "afterwards", "after that", "step", "first", "second",
    "third", "finally", "lastly", "subsequently", "and then", "followed by",
}
TOOL_WORDS = {
    "search", "api", "execute", "query", "call", "invoke", "run", "fetch",
    "lookup", "request", "endpoint", "scrape", "post", "get",
}
CODE_WORDS = {
    "function", "method", "class", "module", "library", "bug", "error",
    "crash", "stack", "trace", "refactor", "restructure", "commit", "merge",
    "rebase", "lint", "compile", "deploy", "endpoint", "schema",
    "implement", "design", "build", "client", "server", "service",
    "queue", "cache", "lock", "thread", "mutex", "test", "code",
    "deployment", "handler", "scheduler", "microservice", "rebuild",
    "migrate", "migration", "package", "repo", "repository", "patch",
    "diagnose", "root cause", "fail", "broken",
    "write", "parse", "validate", "helper", "parser", "script",
    "ingest", "extract", "wrapper",
}
FILE_WORDS = {
    "csv", "tsv", "json", "yaml", "xml", "pdf", "txt", "log", "upload",
    "download", "file", "directory", "folder", "path",
}
COMPARISON = {
    "compare", "comparison", "versus", "vs", "tradeoff", "trade-off",
    "difference", "diff", "contrast",
}

_WORD = re.compile(r"\b[\w'-]+\b")


@dataclass
class Signals:
    word_count: int
    has_multi_step: bool
    has_tool_keywords: bool
    has_code_keywords: bool
    has_file_refs: bool
    has_comparison: bool
    question_count: int
    entity_count: int

    def as_vector(self):
        return [
            self.word_count,
            int(self.has_multi_step),
            int(self.has_tool_keywords),
            int(self.has_code_keywords),
            int(self.has_file_refs),
            int(self.has_comparison),
            self.question_count,
            self.entity_count,
        ]


def _has_any(text_low: str, vocab: set) -> bool:
    return any(w in text_low for w in vocab)


def _count_entities(text: str) -> int:
    tokens = _WORD.findall(text)
    if not tokens:
        return 0
    n = 0
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        if tok[0].isupper() and tok != tok.upper():
            n += 1
    return n


def extract_signals(task: str) -> Signals:
    """Return an 8-feature signal vector for a raw task string.

    Complexity: O(|task|).
    """
    text = task or ""
    low = text.lower()
    return Signals(
        word_count=len(_WORD.findall(text)),
        has_multi_step=_has_any(low, MULTI_STEP),
        has_tool_keywords=_has_any(low, TOOL_WORDS),
        has_code_keywords=_has_any(low, CODE_WORDS),
        has_file_refs=_has_any(low, FILE_WORDS),
        has_comparison=_has_any(low, COMPARISON),
        question_count=text.count("?"),
        entity_count=_count_entities(text),
    )
