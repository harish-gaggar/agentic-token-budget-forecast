"""Render paper.tex to a clean plain-text paper.txt.

This script is a small, opinionated LaTeX-to-text converter scoped to
the constructs that actually appear in paper.tex (sections, paragraphs,
tabular, itemize/enumerate, inline math, \\cite, \\ref, \\code, etc.).
It is not a general LaTeX renderer.

Why we don't use pdftotext: PDF extraction loses paragraph breaks and
brings glyph artefacts (circled numbers, ligatures, table column noise)
that read like AI output even though they came from the typesetter.
Rendering from .tex gives us deterministic, paragraph-correct ASCII.

Output goes to paper.txt next to paper.tex. Run after every paper
edit::

    .venv/bin/python -m experiments.build_paper_txt
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper.tex"
BIB = ROOT / "paper.bib"
OUT = ROOT / "paper.txt"


# --- text-level rewrites ---------------------------------------------------

# Mapping for the small set of unicode characters legitimate in the paper
# that we want to keep as ASCII in the plain-text version.
UNICODE_TO_ASCII = {
    "\u2013": "-",   # en-dash
    "\u2014": ", ",  # em-dash, replaced with ", " so prose still flows
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",
    "\u2212": "-",
}

# LaTeX math sequences that should render as ASCII in plain text.
MATH_REPLACEMENTS = [
    # Command-prefix-safe variants: require a non-letter to follow so
    # that "\to" does not match inside "\toprule" and "\ne" does not
    # match inside "\newcommand", etc.
    (r"\\rightarrow", "->"),
    (r"\\leftarrow", "<-"),
    (r"\\Rightarrow", "=>"),
    (r"\\Leftarrow", "<="),
    (r"\\to(?![a-zA-Z])", "->"),
    (r"\\leq(?![a-zA-Z])", "<="),
    (r"\\geq(?![a-zA-Z])", ">="),
    (r"\\ne(?![a-zA-Z])", "!="),
    (r"\\neq(?![a-zA-Z])", "!="),
    (r"\\approx(?![a-zA-Z])", "~="),
    (r"\\times(?![a-zA-Z])", "x"),
    (r"\\pm(?![a-zA-Z])", "+/-"),
    (r"\\cdot(?![a-zA-Z])", "*"),
    (r"\\max(?![a-zA-Z])", "max"),
    (r"\\min(?![a-zA-Z])", "min"),
    (r"\\sum(?![a-zA-Z])", "sum"),
    (r"\\log(?![a-zA-Z])", "log"),
    (r"\\Pr(?![a-zA-Z])", "Pr"),
    (r"\\,", " "),
    (r"\\;", " "),
    (r"\\:", " "),
    (r"\\!", ""),
    (r"\\quad(?![a-zA-Z])", "  "),
    (r"\\qquad(?![a-zA-Z])", "    "),
    (r"\\dots(?![a-zA-Z])", "..."),
    (r"\\ldots(?![a-zA-Z])", "..."),
    (r"\\%", "%"),
    (r"\\#", "#"),
    # Sentinel for escaped dollar so a subsequent inline-math pass over
    # the same string does not treat it as a $...$ delimiter.
    (r"\\\$", "\x01DOLLAR\x01"),
    (r"\\&", "&"),
    (r"\\_", "_"),
    (r"\\\{", "{"),
    (r"\\\}", "}"),
    # Note: we deliberately do NOT include r"\\\\" here. LaTeX uses
    # "\\\\" as a row separator inside tabular, so stripping it
    # before the table renderer runs would collapse rows. The body
    # and per-cell renderers handle it separately.
]

# Greek and a handful of math symbols. Keep the Greek letters; they
# read as scientific notation, not as decoration.
SYMBOL_REPLACEMENTS = [
    (r"\\alpha", "alpha"),
    (r"\\beta", "beta"),
    (r"\\gamma", "gamma"),
    (r"\\delta", "delta"),
    (r"\\epsilon", "epsilon"),
    (r"\\kappa", "kappa"),
    (r"\\tau", "tau"),
    (r"\\sigma", "sigma"),
    (r"\\mathcal\{O\}", "O"),
    (r"\\mathbb\{R\}", "R"),
    (r"\\hat\{B\}", "B-hat"),
    (r"\\hat\{S\}", "S-hat"),
    (r"\\hat\{T\}", "T-hat"),
    (r"\\hat\{M\}", "M-hat"),
    (r"\\hat\{C\}", "C-hat"),
    (r"\\hat\{R\}", "R-hat"),
    (r"\\hat\{O\}", "O-hat"),
    (r"\\textsc\{[^}]+\}", lambda m: re.sub(r"\\textsc\{|\}", "", m.group(0))),
]


def _strip_unicode(s: str) -> str:
    for src, dst in UNICODE_TO_ASCII.items():
        s = s.replace(src, dst)
    return s


def _apply_replacements(s: str) -> str:
    for pat, repl in MATH_REPLACEMENTS + SYMBOL_REPLACEMENTS:
        s = re.sub(pat, repl, s)
    return s


# --- environment / region handling -----------------------------------------

def _strip_envs(s: str) -> str:
    """Remove TikZ figures and other heavy environments entirely."""
    # Remove whole tikzpicture environments and the figure that wraps them.
    s = re.sub(
        r"\\begin\{figure\}.*?\\end\{figure\}",
        "[Figure omitted; see paper.pdf]",
        s,
        flags=re.DOTALL,
    )
    s = re.sub(
        r"\\begin\{equation\}(.*?)\\end\{equation\}",
        lambda m: "\n\n    " + _math_inline(m.group(1).strip()) + "\n",
        s,
        flags=re.DOTALL,
    )
    return s


def _math_inline(s: str) -> str:
    s = _apply_replacements(s)
    # Strip remaining \command{...} we don't care about.
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    return s.strip()


def _render_table(block: str) -> str:
    """Render a tabular environment to a simple aligned ASCII table."""
    body = re.search(
        r"\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}",
        block,
        flags=re.DOTALL,
    )
    if not body:
        return "[Table omitted; see paper.pdf]"
    rows_raw = body.group(1)
    # Drop \toprule, \midrule, \bottomrule, \\[Npt], \cmidrule, etc.
    rows_raw = re.sub(r"\\(top|mid|bot|cmid)rule(\[[^\]]+\])?", "", rows_raw)
    rows_raw = re.sub(r"\\hline", "", rows_raw)
    rows = []
    for line in rows_raw.split(r"\\"):
        line = line.strip()
        if not line:
            continue
        cells = [_clean_cell(c) for c in line.split("&")]
        if any(c.strip() for c in cells):
            rows.append(cells)
    if not rows:
        return "[Table omitted; see paper.pdf]"
    # Compute column widths.
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    widths = [max(len(r[c]) for r in rows) for c in range(ncols)]
    sep = "  "
    lines = [sep.join(r[c].ljust(widths[c]) for c in range(ncols)).rstrip()
             for r in rows]
    # Caption, if any
    cap = re.search(r"\\caption\{(.+?)\}\s*(\\label|\\end)", block, flags=re.DOTALL)
    out_lines = []
    if cap:
        out_lines.append("[Table] " + _clean_cell(cap.group(1)))
    out_lines.extend(lines)
    return "\n".join(out_lines)


def _clean_cell(s: str) -> str:
    s = _apply_replacements(s)
    s = re.sub(r"\\(textbf|textit|emph|code)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\(small|footnotesize|scriptsize|tiny|normalsize|large)\b\s*", "", s)
    s = re.sub(r"\\(centering|hline|tabularnewline)\b", "", s)
    s = re.sub(r"\$([^$]+)\$", lambda m: _math_inline(m.group(1)), s)
    s = re.sub(r"\\\\(\[[^\]]+\])?", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ").strip()
    s = s.replace("\x01DOLLAR\x01", "$")
    s = re.sub(r"\s+", " ", s)
    return s


# --- the main renderer -----------------------------------------------------

def render(tex: str) -> str:
    # Cut down to the body of the paper (after \begin{document}).
    body_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}",
                           tex, flags=re.DOTALL)
    body = body_match.group(1) if body_match else tex

    # Pull title / abstract from preamble + body.
    title_m = re.search(r"\\title\{([^}]+)\}", tex)
    title = _clean_cell(title_m.group(1)) if title_m else "paper.tex"

    abstract_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                           body, flags=re.DOTALL)
    abstract = abstract_m.group(1) if abstract_m else ""
    # Strip the abstract from body so it doesn't render twice.
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "",
                  body, flags=re.DOTALL)

    body = _strip_unicode(body)
    # Drop \setlength{...}{...} options that show up inside list envs.
    body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", body)
    body = _strip_envs(body)
    # Apply MATH/SYMBOL replacements once to the whole body so escaped
    # punctuation like \%, \_, \&, \, render the same outside math too.
    body = _apply_replacements(body)
    # Resolve \ref / \autoref / \cref labels BEFORE list/table extraction
    # so refs inside list items also render correctly.
    LABEL_TO_NAME = {
        "sec:intro": "Introduction",
        "sec:related": "Related Work",
        "sec:problem": "Problem Formulation",
        "sec:framework": "The TABF Framework",
        "sec:extractor": "Signal Extractor",
        "sec:classifier": "Task Classifier",
        "sec:budget": "Budget Model",
        "sec:setup": "Experimental Setup",
        "sec:results": "Results",
        "sec:optimizer": "ContextOptimizer",
        "sec:pairing": "Pairing with TABF",
        "sec:production": "Production Deployment Case Study",
        "sec:invariants": "Invariant preservation",
        "sec:microbench": "Microbench",
        "sec:repro-prod": "Reproducibility section",
        "sec:discussion": "Discussion",
        "sec:conclusion": "Conclusion",
        "tab:stages": "(context stages)",
        "tab:presets": "(presets)",
        "tab:main": "(main results)",
        "tab:ablation": "(ablation)",
        "tab:tasktype": "(per task type)",
        "tab:prod-savings": "(per-call savings)",
        "tab:prod-cost": "(cost projection)",
        "tab:prod-bq-pop": "(BQ population)",
        "tab:prod-overhead": "(overhead)",
        "tab:prod-layers": "(per layer)",
        "fig:pipeline": "(TABF pipeline)",
        "fig:funnel": "(funnel)",
        "fig:prod-stack": "(deployment)",
        "fig:prod-savings": "(per-call savings)",
        "fig:layers": "(per layer)",
    }

    def _ref_repl(m: re.Match) -> str:
        label = m.group(2)
        mapped = LABEL_TO_NAME.get(label)
        if mapped is None:
            stem = label.split(":", 1)[-1].replace("-", " ").replace("_", " ")
            mapped = f"({stem})"
        return mapped
    body = re.sub(r"\\(ref|autoref|cref|Cref)\{([^}]+)\}", _ref_repl, body)

    # Render tables before stripping commands so we capture the structure.
    table_blocks: list[tuple[str, str]] = []
    def _grab_table(m: re.Match) -> str:
        tok = f"__TABLE_{len(table_blocks)}__"
        table_blocks.append((tok, _render_table(m.group(0))))
        return tok
    body = re.sub(r"\\begin\{table\}.*?\\end\{table\}",
                  _grab_table, body, flags=re.DOTALL)

    # Section / subsection / paragraph headings.
    body = re.sub(r"\\section\*?\{([^}]+)\}",
                  lambda m: f"\n\n## {_clean_cell(m.group(1))}\n", body)
    body = re.sub(r"\\subsection\*?\{([^}]+)\}",
                  lambda m: f"\n\n### {_clean_cell(m.group(1))}\n", body)
    # Paragraph heading: render as its own line in bold-like uppercase
    # of the first word, but without markdown emphasis markers (those
    # read as machine-generated formatting in plain text).
    body = re.sub(r"\\paragraph\{([^}]+)\}",
                  lambda m: f"\n\n{_clean_cell(m.group(1)).rstrip('.')}. ", body)

    def _render_list(inner: str, ordered: bool) -> str:
        # Drop any leading \setlength{...}{...} option block.
        inner = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", inner)
        items = re.split(r"\\item\b", inner)
        items = [it.strip() for it in items if it.strip()]
        lines = []
        for i, it in enumerate(items, 1):
            marker = f"{i}." if ordered else "-"
            lines.append(f"  {marker} {_clean_cell(it)}")
        # Wrap in blank lines so the paragraph reflow keeps newlines.
        return "\n\n" + "\n".join(lines) + "\n\n"

    body = re.sub(
        r"\\begin\{itemize\}(?:\[[^\]]*\])?(.*?)\\end\{itemize\}",
        lambda m: _render_list(m.group(1), ordered=False),
        body, flags=re.DOTALL,
    )
    body = re.sub(
        r"\\begin\{enumerate\}(?:\[[^\]]*\])?(.*?)\\end\{enumerate\}",
        lambda m: _render_list(m.group(1), ordered=True),
        body, flags=re.DOTALL,
    )

    # Inline math. The escaped-dollar sentinel was inserted earlier by
    # _apply_replacements, so $...$ here only matches actual math.
    body = re.sub(r"\$([^$]+)\$", lambda m: _math_inline(m.group(1)), body)
    body = body.replace("\x01DOLLAR\x01", "$")

    # LaTeX double / single quotes -> ASCII straight quotes.
    body = re.sub(r"``([^`'\n]*)''", r'"\1"', body)
    body = re.sub(r"`([^`'\n]+)'", r"'\1'", body)

    # \cite / \citep / \citet -> short bracketed token (we keep the
    # bibtex key; the plain text isn't trying to be a publication ref).
    body = re.sub(r"\\cite[tp]?\{([^}]+)\}",
                  lambda m: "[" + m.group(1).replace(",", "; ") + "]", body)

    # Common single-arg formatting commands -> just the argument.
    body = re.sub(r"\\(textbf|textit|emph|code|texttt)\{([^}]*)\}", r"\2", body)
    body = re.sub(r"\\(label|index)\{[^}]+\}", "", body)
    body = re.sub(r"\\(small|footnotesize|scriptsize|tiny|normalsize|large"
                  r"|centering|hline|tabularnewline|bibliography(?:style)?)\b"
                  r"\{?[^}]*\}?", "", body)
    body = re.sub(r"\\(begin|end)\{[a-zA-Z*]+\}\s*(\[[^\]]*\])?", "", body)

    # Final cleanup of stray commands.
    body = re.sub(r"\\[a-zA-Z]+\*?\s*", "", body)
    body = body.replace("{", "").replace("}", "").replace("~", " ")
    body = body.replace("\\\\", " ")
    # LaTeX en-dash "--" -> ASCII hyphen. Numeric ranges and compound
    # names both read naturally with a single hyphen in plain text.
    body = body.replace("--", "-")
    body = body.replace("\x01DOLLAR\x01", "$")

    # Put tables back.
    for tok, rendered in table_blocks:
        body = body.replace(tok, "\n\n" + rendered + "\n")

    # Reflow paragraphs: split on blank lines, collapse internal whitespace.
    # Preserve newlines inside list, table, and figure blocks.
    paras = re.split(r"\n\s*\n", body)
    cleaned = []
    # A paragraph is preformatted if ANY of its non-empty lines starts
    # with a heading marker, a table marker, or a list marker.
    list_marker_re = re.compile(r"^\s*(?:-|\d+\.)\s")
    pre_prefixes = ("##", "###", "[Table]", "[Figure")
    for p in paras:
        p = p.strip("\n").rstrip()
        if not p.strip():
            continue
        nonempty = [ln for ln in p.split("\n") if ln.strip()]
        is_pre = any(
            ln.lstrip().startswith(pre_prefixes) or list_marker_re.match(ln)
            for ln in nonempty
        )
        if is_pre:
            cleaned.append("\n".join(ln.rstrip() for ln in nonempty))
        else:
            cleaned.append(re.sub(r"\s+", " ", p))

    abstract_txt = _clean_cell(abstract)
    header = (
        f"{title}\n\n"
        f"Abstract\n\n"
        f"{abstract_txt}\n"
    )

    return header + "\n\n" + "\n\n".join(cleaned) + "\n"


def _render_bib(bib_text: str) -> str:
    """Render paper.bib into a plain-text References section.

    Mirrors what bibtex would put at the end of the PDF, but as ASCII
    so paper.txt is a self-contained reviewable artefact. Uses the same
    entries that paper.tex cites, sorted by first-author surname.
    """
    entries = re.findall(
        r"@(\w+)\s*\{\s*([^,]+)\s*,\s*(.*?)\n\}",
        bib_text,
        flags=re.DOTALL,
    )

    def _field(body: str, key: str) -> str:
        m = re.search(rf"{key}\s*=\s*\{{(.*?)\}}\s*,?\s*\n", body, flags=re.DOTALL)
        if not m:
            return ""
        v = m.group(1)
        v = re.sub(r"\\[A-Za-z]+\{([^}]*)\}", r"\1", v)  # \textit{x} -> x
        v = re.sub(r"\\['`\"^~={.][A-Za-z]", lambda mm: mm.group(0)[-1], v)  # \'e -> e
        for src, dst in [(r"{\.I}", "I"), (r"{\.i}", "i"), (r"{\\i}", "i"),
                         (r"{\\j}", "j")]:
            v = v.replace(src, dst)
        v = v.replace("{", "").replace("}", "")
        v = re.sub(r"\s+", " ", v).strip().rstrip(",")
        return v

    def _format_authors(raw: str) -> str:
        if not raw:
            return ""
        parts = [a.strip() for a in raw.split(" and ")]
        out = []
        for p in parts:
            if p == "others":
                out.append("et al.")
                continue
            if "," in p:
                last, first = [s.strip() for s in p.split(",", 1)]
                out.append(f"{first} {last}")
            else:
                out.append(p)
        if len(out) == 1:
            return out[0]
        if len(out) == 2:
            return f"{out[0]} and {out[1]}"
        return ", ".join(out[:-1]) + ", and " + out[-1]

    rendered = []
    for etype, key, body in entries:
        authors = _format_authors(_field(body, "author"))
        title = _field(body, "title")
        year = _field(body, "year")
        journal = _field(body, "journal")
        institution = _field(body, "institution")
        url = _field(body, "url")
        note = _field(body, "note")
        venue = journal or (f"Technical report, {institution}" if institution else "")
        line = f"{authors}. {title}."
        if venue:
            line += f" {venue},"
        if year:
            line += f" {year}."
        if url:
            line += f" URL {url}."
        if note:
            line += f" {note}."
        # Sort key: first author surname (last token of first author).
        first_author = authors.split(",", 1)[0].split(" and ", 1)[0]
        sort_key = first_author.split()[-1].lower() if first_author else key
        rendered.append((sort_key, line))

    rendered.sort(key=lambda x: x[0])
    return "## References\n\n" + "\n\n".join(line for _, line in rendered) + "\n"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} does not exist.")
    tex = SRC.read_text(encoding="utf-8")
    txt = render(tex)
    if BIB.exists():
        txt = txt.rstrip() + "\n\n" + _render_bib(BIB.read_text(encoding="utf-8"))
    OUT.write_text(txt, encoding="utf-8")
    print(f"Wrote {OUT}  ({len(txt):,} chars)")


if __name__ == "__main__":
    main()
