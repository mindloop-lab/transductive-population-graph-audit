#!/usr/bin/env bash
set -euo pipefail

# Independent secret scan for the tracked release tree.
# Version and archive digest are pinned so CI does not silently change scanner behavior.
GITLEAKS_VERSION="8.30.1"
GITLEAKS_SHA256="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
ARCHIVE="gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
URL="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/${ARCHIVE}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsSL "$URL" -o "$tmp/$ARCHIVE"
printf '%s  %s\n' "$GITLEAKS_SHA256" "$tmp/$ARCHIVE" | sha256sum -c -
tar -xzf "$tmp/$ARCHIVE" -C "$tmp" gitleaks

# Scan an export of Git-tracked files only. The private audit repository intentionally
# retains old development history; the eventual clean-root repository will additionally
# run a full-history gitleaks scan.
mkdir "$tmp/tree"
git archive --format=tar HEAD | tar -xf - -C "$tmp/tree"
"$tmp/gitleaks" dir --no-banner --redact --exit-code 1 "$tmp/tree"

echo "GITLEAKS PASS: no secrets detected in the tracked release tree"
