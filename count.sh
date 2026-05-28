#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder>" >&2
    exit 1
fi

folder="$1"

if [ ! -d "$folder" ]; then
    echo "Error: '$folder' is not a directory" >&2
    exit 1
fi

find "$folder" -maxdepth 1 -type f -iname '*.png' | wc -l
