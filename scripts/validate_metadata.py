#!/usr/bin/env python3
"""Validate the release metadata files.

Both `.zenodo.json` and `CITATION.cff` feed the archived record, and `.zenodo.json`
takes precedence during GitHub release archiving. A syntax error in either one is a
release defect, so this gate parses both and checks they agree.

Run:  python scripts/validate_metadata.py
Exit: 0 clean, 1 on any violation.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZEN = ROOT / ".zenodo.json"
CFF = ROOT / "CITATION.cff"

errors: list[str] = []

# ---- .zenodo.json -------------------------------------------------------------
zen = {}
if not ZEN.is_file():
    errors.append(".zenodo.json is missing")
else:
    try:
        zen = json.loads(ZEN.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f".zenodo.json is not valid JSON: {exc}")

if zen:
    lic = zen.get("license")
    if not isinstance(lic, str) or not lic:
        errors.append(f".zenodo.json license must be a string (got {lic!r})")
    if not zen.get("creators"):
        errors.append(".zenodo.json has no creators")
    if zen.get("upload_type") not in ("software", "dataset", "publication", None):
        errors.append(f".zenodo.json upload_type looks wrong: {zen.get('upload_type')!r}")
    for field in ("title", "description"):
        if not zen.get(field):
            errors.append(f".zenodo.json has an empty {field}")

# ---- CITATION.cff -------------------------------------------------------------
cff = {}
if not CFF.is_file():
    errors.append("CITATION.cff is missing")
else:
    text = CFF.read_text()
    if '\\"' in text:
        errors.append('CITATION.cff contains escaped quotes (\\") -- invalid YAML')
    try:
        import yaml  # type: ignore
        cff = yaml.safe_load(text) or {}
    except ImportError:
        print("note: PyYAML unavailable; checking CITATION.cff structurally only")
        for key in ("cff-version:", "title:", "version:", "license:", "repository-code:"):
            if key not in text:
                errors.append(f"CITATION.cff is missing {key}")
    except Exception as exc:  # noqa: BLE001 - surface any parse failure
        errors.append(f"CITATION.cff is not valid YAML: {exc}")

if cff:
    if not cff.get("version"):
        errors.append("CITATION.cff has no version")
    if not cff.get("license"):
        errors.append("CITATION.cff has no license")
    if not cff.get("authors"):
        errors.append("CITATION.cff has no authors")

# ---- agreement ----------------------------------------------------------------
if zen and cff:
    zl = str(zen.get("license", "")).lower().replace("-", "").replace(".", "").replace("_", "")
    cl = str(cff.get("license", "")).lower().replace("-", "").replace(".", "").replace("_", "")
    if zl and cl and zl != cl:
        errors.append(f"licence disagreement: .zenodo.json={zen.get('license')!r} "
                      f"CITATION.cff={cff.get('license')!r}")
    zv, cv = str(zen.get("version", "")), str(cff.get("version", ""))
    if zv and cv and zv != cv:
        errors.append(f"version disagreement: .zenodo.json={zv!r} CITATION.cff={cv!r}")

if errors:
    print(f"FAIL: {len(errors)} metadata violation(s)")
    for e in errors[:50]:
        print("  -", e)
    sys.exit(1)
print("PASS: .zenodo.json and CITATION.cff parse and agree")
