#!/usr/bin/env python3
"""Audit the tracked public tree for machine-local, secret-like and AI-tool traces.

FORBIDDEN hits fail the run. REVIEW hits (AI/agent tool names) are printed for human
judgement and do not fail.

Only Git-tracked text files are scanned, and the run FAILS if it could not scan a
plausible number of them: a hygiene gate that silently inspects nothing would otherwise
report PASS, which is worse than having no gate at all.

Run:  python scripts/validate_public_tree.py
Exit: 0 clean, 1 on any FORBIDDEN hit or if the audit could not inspect the tree.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

FORBIDDEN = {
    "unix_machine_path": re.compile(r"(?:/mnt/|/Users/|/home/)[^\s\"'<>()\[\]{}]*"),
    "file_uri": re.compile(r"file:///[^\s\"'<>()\[\]{}]+"),
    "internal_workspace_id": re.compile(r"(?i)\b(?:2609-wan|Wan-2608|hpz8g4|king-Z4G4)\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}

REVIEW = {
    "ai_tool_name": re.compile(
        r"(?i)\b(?:ChatGPT|OpenAI|Hermes|Codex|DeepSeek|Claude|Gemini|GPT[- ]?\d(?:\.\d+)?)\b"),
}

# This validator contains the signatures it searches for, so it excludes itself.
SKIP_PATHS = {pathlib.PurePosixPath("scripts/validate_public_tree.py")}


def tracked_files() -> list[pathlib.Path]:
    """Tracked files, split on the real NUL byte that `git ls-files -z` emits.

    Falls back to a filesystem walk when git is unavailable (e.g. a tar-extracted copy),
    so the audit still inspects something instead of crashing or silently passing.
    """
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL)
        files = [ROOT / p.decode("utf-8") for p in out.split(b"\x00") if p]
        if files:
            return files
    except (OSError, subprocess.CalledProcessError):
        pass
    skip_dirs = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    return [p for p in sorted(ROOT.rglob("*"))
            if p.is_file() and not (skip_dirs & set(p.relative_to(ROOT).parts))
            and p.suffix not in {".pyc", ".pyo"}]


def line_no(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


paths = tracked_files()
forbidden_hits: list[str] = []
review_hits: list[str] = []
scanned = skipped = 0

for path in paths:
    try:
        rel = pathlib.PurePosixPath(path.relative_to(ROOT).as_posix())
    except ValueError:
        rel = pathlib.PurePosixPath(path.name)
    if rel in SKIP_PATHS or not path.is_file():
        continue
    raw = path.read_bytes()
    if b"\x00" in raw:
        skipped += 1
        continue
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        skipped += 1
        continue
    scanned += 1
    for name, pattern in FORBIDDEN.items():
        for match in pattern.finditer(text):
            forbidden_hits.append(
                f"{rel}:{line_no(text, match.start())}: {name}: {match.group(0)[:160]}")
    for name, pattern in REVIEW.items():
        for match in pattern.finditer(text):
            review_hits.append(f"{rel}:{line_no(text, match.start())}: {name}: {match.group(0)}")

print(f"public-tree audit: tracked={len(paths)}, scanned={scanned}, "
      f"binary/non-UTF8 skipped={skipped}")

# A gate that scanned nothing cannot vouch for anything.
if scanned < 0.5 * len(paths):
    print(f"FAIL: the audit inspected only {scanned} of {len(paths)} tracked files; "
          f"it cannot vouch for the tree")
    sys.exit(1)

print(f"FORBIDDEN hits: {len(forbidden_hits)}")
for hit in forbidden_hits[:100]:
    print("  -", hit)
print(f"REVIEW hits: {len(review_hits)}")
for hit in review_hits[:100]:
    print("  -", hit)
if forbidden_hits:
    print("FAIL: tracked public tree contains machine-local or credential-like material")
    sys.exit(1)
print("PASS: no forbidden machine-local or credential-like material in tracked public tree")
