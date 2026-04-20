#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DL_SCRIPT="$SCRIPT_DIR/download_kurosiwo.sh"

mapfile -t urls < <(grep -oE 'https://www\.dropbox\.com/[^"]+' "$DL_SCRIPT")

labels=(catalogue.gpkg 00.tar.gz 01.tar.gz 02.tar.gz 03.tar.gz 04.tar.gz 05.tar.gz 06.tar.gz 07.tar.gz 08.tar.gz 09.tar.gz 10.tar.gz)

total=0
printf "%-18s %15s  %s\n" "FILE" "BYTES" "HUMAN"
for i in "${!urls[@]}"; do
    u="${urls[i]//dl=0/dl=1}"
    size=$(curl -sIL "$u" | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {v=$2} END{gsub(/\r/,"",v); print v+0}')
    human=$(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "$size")
    printf "%-18s %15s  %s\n" "${labels[i]}" "$size" "$human"
    total=$((total + size))
done
echo "----------------------------------------"
printf "%-18s %15s  %s\n" "TOTAL" "$total" "$(numfmt --to=iec --suffix=B "$total")"
