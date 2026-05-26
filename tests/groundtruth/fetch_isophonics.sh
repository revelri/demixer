#!/usr/bin/env bash
# Fetch Isophonics reference annotations (Beatles + Queen) and vendor the
# chord/key/beat label files under tests/groundtruth/isophonics/.
#
# Source: http://isophonics.net/content/reference-annotations  (CC-licensed,
# human-annotated). These are the ground truth for demixer.eval.groundtruth_matrix.
# Run once; the .lab/.txt files are committed so the eval needs no network.
set -euo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)/isophonics"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
for name in "The Beatles" "Queen"; do
  url="http://isophonics.net/files/annotations/${name// /%20} Annotations.tar.gz"
  url="${url// /%20}"
  echo "fetching $name ..."
  curl -fsSL --max-time 120 -o ann.tar.gz "$url"
  tar xzf ann.tar.gz && rm ann.tar.gz
done
# keep only the label kinds the matrix scores; drop seglab/all and RDF .ttl
for kind in chordlab keylab beat; do
  find "$kind" -type f \( -name '*.lab' -o -name '*.txt' \) -print0 2>/dev/null \
    | while IFS= read -r -d '' f; do mkdir -p "$DEST/$(dirname "$f")"; cp "$f" "$DEST/$f"; done
done
echo "vendored: $(find "$DEST" -type f | wc -l) files under $DEST"
