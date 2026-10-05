"""Checks that the documentation stays in sync with the code (issues #13 and #15).

- The study notes cite code as `function`, `file.py:N`. Line N must lie inside that function:
  line numbers go stale whenever code is inserted above them. Names that are not functions
  of that file (constants such as DIVERGING) are skipped.
- Every line citation (`file.py:N`, a range `file.py:N–M`, or the short form `:N` for the
  file cited just before it on the same line) must name the statement it points to, in a
  hidden comment right after it: `fairing/fairing.py:33`<!-- A = A @ D @ M -->. The
  statement must be on the cited line (or in the range). Citing a function's own `def`
  line, right after its backticked name, needs no comment (#33).
- Every figure in docs/img/ must be written by an example script that the README lists, and
  shown in the README gallery and in a phase note.
- Relative links and images in the README and the notes must point to existing files.
- The README's quick-start code must run.
"""

import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
NOTES = sorted((ROOT / "docs" / "phases").glob("*.md"))
# a backticked name, then (within the same table cell / phrase) a backticked file:line
CITATION = re.compile(r"`(\w+)`(?:[^`|;]{0,25})`(?:fairing/)?(\w+\.py):(\d+)`")


def citations():
    for note in NOTES:
        for match in CITATION.finditer(note.read_text()):
            yield note.name, *match.groups()


def test_notes_contain_function_citations():
    assert len(list(citations())) > 10


def function_span(source, name):
    """(first, last) line numbers of a top-level function, or None if the file has no such def."""
    starts = [i for i, line in enumerate(source, 1) if line.startswith(("def ", "class "))]
    for i, start in enumerate(starts):
        if source[start - 1].startswith(f"def {name}("):
            end = starts[i + 1] - 1 if i + 1 < len(starts) else len(source)
            return start, end
    return None


def test_function_citations_point_inside_the_function():
    stale = []
    for note, name, filename, line in citations():
        span = function_span((ROOT / "fairing" / filename).read_text().splitlines(), name)
        if span is not None and not span[0] <= int(line) <= span[1]:
            stale.append(f"{note}: `{name}` cited at {filename}:{line}, but it spans lines {span[0]}-{span[1]}")
    assert not stale, "stale line references:\n" + "\n".join(stale)


# a backticked file:line, file:first–last or :line, optionally followed by <!-- statement -->
LINE_CITATION = re.compile(r"`(?:(?:fairing/)?(\w+\.py))?:(\d+)(?:[–-](\d+))?`(?:<!--(.*?)-->)?")


def line_citations():
    """Every line citation in the notes, as (where, file, first, last, statement, name).

    ``statement`` is the text of the hidden comment after the citation (None if there is
    none); ``name`` is the backticked function cited just before it, if any.
    """
    for note in NOTES:
        for line_no, line in enumerate(note.read_text().splitlines(), 1):
            names = {(filename, int(n)): name for name, filename, n in CITATION.findall(line)}
            filename = None
            for match in LINE_CITATION.finditer(line):
                filename = match.group(1) or filename          # `:N` reuses the previous file
                first = int(match.group(2))
                last = int(match.group(3) or first)
                statement = match.group(4).strip() if match.group(4) else None
                yield (f"{note.name}:{line_no}", filename, first, last, statement,
                       names.get((filename, first)))


def test_notes_contain_line_citations():
    assert len(list(line_citations())) > 40


def test_cited_lines_contain_their_statement():
    problems = []
    for where, filename, first, last, statement, name in line_citations():
        cite = f"{where}, `{filename}:{first}{f'–{last}' if last != first else ''}`"
        if filename is None:
            problems.append(f"{cite}: a short `:N` citation needs a `file.py:N` before it on the same line")
            continue
        source = (ROOT / "fairing" / filename).read_text().splitlines()
        cited = source[first - 1:last]
        if statement is None:
            if not (name and cited and cited[0].startswith(f"def {name}(")):
                problems.append(f"{cite}: name the cited statement in <!-- ... --> right after it")
        elif not any(statement in text for text in cited):
            now = [i for i, text in enumerate(source, 1) if statement in text]
            problems.append(f"{cite}: `{statement}` is not there; "
                            + (f"it is now on line {', '.join(map(str, now))}" if now else "it is no longer in the file"))
    assert not problems, "citations that do not point at their statement:\n" + "\n".join(problems)


def test_every_figure_is_made_by_a_listed_example_and_shown():
    scripts = [p for p in sorted((ROOT / "examples").glob("*.py")) if "polyscope" not in p.name]
    notes = "\n".join(note.read_text() for note in NOTES)
    figures = sorted((ROOT / "docs" / "img").glob("*.png"))
    assert figures
    problems = []
    for png in figures:
        makers = [p for p in scripts if f'"{png.name}"' in p.read_text()]
        if not makers:
            problems.append(f"{png.name}: no example script writes it")
        problems += [f"{png.name}: {p.name} is not in the README" for p in makers
                     if f"examples/{p.name}" not in README]
        if f"docs/img/{png.name}" not in README:
            problems.append(f"{png.name}: not in the README gallery")
        if f"../img/{png.name}" not in notes:
            problems.append(f"{png.name}: not shown in any phase note")
    assert not problems, "\n".join(problems)


LINK = re.compile(r"\]\(([^)\s]+)\)|src=\"([^\"]+)\"")


def test_relative_links_point_to_existing_files():
    documents = [ROOT / "README.md", ROOT / "docs" / "PROGRESS.md", *NOTES]
    broken = []
    for doc in documents:
        for match in LINK.finditer(doc.read_text()):
            target = (match.group(1) or match.group(2)).split("#")[0]
            if target and not target.startswith(("http://", "https://")) and not (doc.parent / target).exists():
                broken.append(f"{doc.relative_to(ROOT)}: {target}")
    assert not broken, "broken links:\n" + "\n".join(broken)


def test_readme_quick_start_runs(monkeypatch):
    section = README.split("## Quick start", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    monkeypatch.setattr(plt, "show", lambda: None)
    exec(compile(code, "README.md (Quick start)", "exec"), {})
    assert plt.get_fignums()
    plt.close("all")
