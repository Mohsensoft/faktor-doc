#!/usr/bin/env python3
"""
Generates a JSONL dataset optimized for text embeddings from plain-text files.

- Reads all .txt files in the ./text directory (UTF-8).
- Cleans and normalizes whitespace.
- Splits content into overlapping chunks for robust embeddings.
- Emits a JSONL file (embeddings.jsonl) with one record per line.

Each record has:
{
  "id": "<stable-unique-id>",
  "url": "https://mohsensoft.com/docs/faktor/<stem>.html",
  "title": "<derived from filename or first heading>",
  "chunk_index": <0-based>,
  "text": "<chunk content>"
}
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

WORKSPACE_ROOT = Path(__file__).parent.resolve()
TEXT_DIR = WORKSPACE_ROOT / "text"
OUTPUT_PATH = WORKSPACE_ROOT / "embeddings.jsonl"

# Chunking defaults tuned for common embedding models (approx. 700-1100 tokens)
DEFAULT_CHUNK_SIZE = 1400  # characters
DEFAULT_CHUNK_OVERLAP = 200  # characters


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    url: str
    title: str
    chunk_index: int
    text: str


def read_text_file(path: Path) -> str:
    """
    Read a text file as UTF-8 (with BOM tolerance). Falls back to cp1256/latin encodings
    if necessary, prioritizing Persian-friendly decoding.
    """
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        for enc in ("cp1256", "windows-1256", "iso-8859-1"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
    # If all decodings fail, re-raise with context
    return path.read_text(encoding="utf-8")


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph and list structure.
    - Convert Windows newlines to \n
    - Collapse 3+ blank lines to 2
    - Trim trailing spaces
    - Normalize internal spaces
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Replace tabs with two spaces to stabilize layout without expanding too far
    text = text.replace("\t", "  ")
    # Collapse long runs of blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces inside lines (but keep indentation-like leading spaces)
    def _collapse_internal_spaces(line: str) -> str:
        if not line:
            return line
        # Preserve leading spaces indentation, normalize the rest
        leading = len(line) - len(line.lstrip(" "))
        head = line[:leading]
        tail = re.sub(r"[ ]{2,}", " ", line[leading:])
        return head + tail
    text = "\n".join(_collapse_internal_spaces(line) for line in text.split("\n"))
    return text.strip()


def strip_sphinx_markup(text: str) -> str:
    """
    Remove common Sphinx/reStructuredText markups while keeping readable content.
    - Drop adornment lines (====, ----, ~~~, ****, etc.)
    - Handle overline+underline headings by keeping the title line only
    - Strip Sphinx roles: :role:`text` -> text
    - Simplify links: `label <url>`_ -> label, `name`_ -> name
    - Unwrap inline literals/emphasis: ``code``/**bold**/*italic* -> plain
    - Drop directive lines like: .. note::, .. code-block:: (but keep following content)
    - Replace substitutions: |name| -> name
    - Remove bracket tags like [label][ref] or [label]
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: List[str] = []
    adorn_re = re.compile(r"[=\-~^`:'\"*+#_.]{3,}\s*$")
    directive_re = re.compile(r"\.\.\s+[\w-]+::")

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Overline + title + underline (same char)
        if adorn_re.fullmatch(line or ""):
            ch = line.strip()[0] if line.strip() else ""
            if ch and i + 2 < n and adorn_re.fullmatch(lines[i + 2] or ""):
                if lines[i + 2].strip()[0] == ch:
                    # Keep the middle line as title, skip both adornment lines
                    out.append(lines[i + 1])
                    i += 3
                    continue
            # Otherwise, treat as simple underline for previous title: skip
            i += 1
            continue

        # Directive line: remove it, keep subsequent content
        if directive_re.match(line.strip()):
            i += 1
            continue

        s = line
        # Roles :role:`text`
        s = re.sub(r":\w+:`([^`]+)`", r"\1", s)
        # Links `label <url>`_
        s = re.sub(r"`([^`<>]+?)\s*<[^`>]+>`_", r"\1", s)
        # Named refs `name`_
        s = re.sub(r"`([^`]+)`_", r"\1", s)
        # Inline literals ``code``
        s = re.sub(r"``([^`]+)``", r"\1", s)
        # Bold/italic
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
        s = re.sub(r"\*([^*\s][^*]*?)\*", r"\1", s)
        # Substitutions |name|
        s = re.sub(r"\|([^|]+)\|", r"\1", s)
        # Bracket tags like [label][ref] and standalone [label]
        s = re.sub(r"\[[^\[\]\n]+\]\[[^\[\]\n]+\]", "", s)
        s = re.sub(r"\[[^\[\]\n]+\]", "", s)

        out.append(s)
        i += 1

    # Remove standalone adornment lines that might remain
    out = [ln for ln in out if not adorn_re.fullmatch(ln or "")]
    return "\n".join(out)


def derive_title(filename: str, content: str) -> str:
    """
    Derive a human-friendly title. Prefer the first non-empty line if it looks like a heading,
    otherwise use the filename (without extension).
    """
    stem = Path(filename).stem
    first_non_empty = next((ln.strip() for ln in content.split("\n") if ln.strip()), "")
    if not first_non_empty:
        return stem
    # If the first line is short or looks like a header, use it
    if len(first_non_empty) <= 80:
        return first_non_empty
    return stem


def stable_id(*parts: str) -> str:
    """
    Create a short stable id from parts using SHA-1 (sufficient for dataset ids).
    """
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks by character length.
    """
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        # Expand to nearest paragraph boundary if possible (look forward)
        if end < n:
            next_para = text.find("\n\n", end, min(n, end + 200))
            if next_para != -1:
                end = next_para + 2
                chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= n:
            break
        start = max(end - overlap, 0)
    # Filter empty chunks
    return [c for c in chunks if c]


def iter_text_files(root: Path) -> Iterable[Path]:
    skip_names = {"changes-log.txt", "index.txt"}
    for p in sorted(root.glob("*.txt")):
        if p.is_file():
            if p.name in skip_names:
                continue
            yield p


def build_records() -> List[ChunkRecord]:
    if not TEXT_DIR.exists():
        raise FileNotFoundError(f"Text directory not found: {TEXT_DIR}")
    records: List[ChunkRecord] = []
    for path in iter_text_files(TEXT_DIR):
        raw = read_text_file(path)
        # First, remove Sphinx/reST markups, then normalize whitespace
        cleaned = normalize_whitespace(strip_sphinx_markup(raw))
        title = derive_title(path.name, cleaned)
        chunks = chunk_text(cleaned)
        # Guarantee at least one chunk (even if file is empty after cleaning)
        if not chunks:
            chunks = [""]
        stem = Path(path.name).stem
        url = f"https://mohsensoft.com/docs/faktor/{stem}.html"
        base_id = stable_id(url, title)
        for idx, chunk in enumerate(chunks):
            rec = ChunkRecord(
                id=stable_id(base_id, str(idx)),
                url=url,
                title=title,
                chunk_index=idx,
                text=chunk,
            )
            records.append(rec)
    return records


def write_json(records: List[ChunkRecord], out_path: Path) -> None:
    # Write JSON Lines: one compact JSON object per line
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            obj = dict(id=r.id, url=r.url, title=r.title, chunk_index=r.chunk_index, text=r.text)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    records = build_records()
    write_json(records, OUTPUT_PATH)
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


