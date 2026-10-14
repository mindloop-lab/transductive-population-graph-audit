#!/usr/bin/env python3
"""Build the R3 evidence pack. Run this AFTER every record has been finalised (results doc included).

Members: every top-level workspace record, the run outputs, the orientation diagnostic, the vector
figures, and the source+PDF rebuilt from the frozen baseline source. Members are de-duplicated by
archive name and the checksum list excludes itself.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

PAPERM = Path("/mnt/disk_8tb/codex/2609-wan/paperM")
R3 = PAPERM / "PAPERM_CMIG_R3_20261011"
NAME = "PAPERM_CMIG_R3_EVIDENCE_PACK"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


members: list[tuple[Path, str]] = []
for pat in ("*.md", "*.json", "*.csv", "*.py", "*.yaml", "*.sh", "*.log", "*.txt", "*.json5"):
    for p in sorted(R3.glob(pat)):
        members.append((p, p.name))
for sub, pat in (("major1_runs", "*_report.json"), ("major1_runs", "*_oof.npz"),
                 ("major1_runs", "smoke/*"), ("major1_runs_orientation", "*_report.json"),
                 ("major1_runs_orientation", "*_oof.npz"),
                 ("major1_runs_sym", "*_report.json"), ("major1_runs_sym", "*_oof.npz"),
                 ("P1_support_runs", "*_report.json"), ("P1_support_runs", "*_oof.npz"),
                 ("r3_figures", "*"),
                 ("R3_pdf_rebuild", "*.tex"), ("R3_pdf_rebuild", "*.pdf"),
                 ("R3_revision/source", "*.tex"), ("R3_revision/source", "*.pdf"),
                 ("R3_revision/source", "*.log")):
    for p in sorted((R3 / sub).glob(pat)):
        if p.is_file():
            members.append((p, f"{sub}/{p.name}"))

seen: set[str] = set()
uniq: list[tuple[Path, str]] = []
for p, arc in members:
    if arc in seen:
        print(f"  [dedupe] {arc}")
        continue
    seen.add(arc)
    uniq.append((p, arc))

sums = "\n".join(sorted(f"{sha(p)}  {arc}" for p, arc in uniq)) + "\n"
zp = R3 / f"{NAME}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p, arc in uniq:
        z.write(p, arc)
    z.writestr("SHA256SUMS.txt", sums)
(R3 / f"{NAME}.zip.sha256").write_text(f"{sha(zp)}  {zp.name}\n", encoding="utf-8")
print(f"pack {zp.name}: {zp.stat().st_size} B, {len(uniq)} members, sha256 {sha(zp)}")
