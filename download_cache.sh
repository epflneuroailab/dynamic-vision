#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
URL="https://epfl-neuroailab-public.s3.amazonaws.com/david/cache.tar.gz"
CACHE_DIR="cache"
mkdir -p "$CACHE_DIR"
cd "$CACHE_DIR"

FILENAME="$(basename "$URL")"

# --- Download ---
echo "Checking for parallel curl support..."
if curl --help | grep -q -- "--parallel"; then
    echo "✅ Using parallel curl download..."
    curl --parallel -L -O "$URL"
else
    echo "⚙️ Parallel mode not supported — using normal curl."
    curl -L -O "$URL"
fi

# --- Extract ---
echo "Extracting $FILENAME..."
tar -xzf "$FILENAME"

# --- Cleanup ---
echo "Removing $FILENAME..."
rm -f "$FILENAME"

echo "✅ Done. Extracted into $CACHE_DIR/"
