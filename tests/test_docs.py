"""The study notes cite code as `function`, `file.py:N`. Check that line N lies inside that function.

Line numbers go stale whenever code is inserted above them; this test catches it.
Names that are not functions of that file (constants such as DIVERGING) are skipped.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
# a backticked name, then (within the same table cell / phrase) a backticked file:line
CITATION = re.compile(r"`(\w+)`(?:[^`|;]{0,25})`(?:fairing/)?(\w+\.py):(\d+)`")


def citations():
    for note in sorted((ROOT / "docs" / "phases").glob("*.md")):
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
