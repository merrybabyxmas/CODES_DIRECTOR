#!/usr/bin/env bash
# ============================================================================
# download_trailers.sh
#
# Downloads movie trailers from YouTube using yt-dlp for DIRECTOR pipeline.
#
# Usage:
#   bash scripts/download_trailers.sh [--output_dir DIR] [--list FILE] [--max N]
#
# The script reads YouTube URLs from a text file (one per line) and downloads
# each video at 720p (or best available below 1080p) in mp4 format.
#
# Default URL list: scripts/trailer_urls.txt
# Default output:   data/raw_videos/
# ============================================================================

set -euo pipefail

# ---- Defaults ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/data/raw_videos"
URL_LIST="${SCRIPT_DIR}/trailer_urls.txt"
MAX_DOWNLOADS=0  # 0 = unlimited
COOKIES=""
RATE_LIMIT="5M"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --list)       URL_LIST="$2";   shift 2 ;;
        --max)        MAX_DOWNLOADS="$2"; shift 2 ;;
        --cookies)    COOKIES="$2"; shift 2 ;;
        --rate_limit) RATE_LIMIT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--output_dir DIR] [--list FILE] [--max N] [--cookies FILE] [--rate_limit RATE]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---- Check dependencies ----
if ! command -v yt-dlp &>/dev/null; then
    echo "ERROR: yt-dlp not found. Install with: pip install yt-dlp"
    exit 1
fi

# ---- Create output dir ----
mkdir -p "$OUTPUT_DIR"

# ---- Check URL list ----
if [[ ! -f "$URL_LIST" ]]; then
    echo "URL list not found at: $URL_LIST"
    echo "Creating a sample file with example trailer URLs..."
    cat > "$URL_LIST" << 'URLS'
# DIRECTOR Dataset - Trailer URLs
# One YouTube URL per line. Lines starting with # are ignored.
# Add movie trailer URLs below:
#
# Example (replace with actual URLs you have rights to use):
# https://www.youtube.com/watch?v=EXAMPLE_ID_1
# https://www.youtube.com/watch?v=EXAMPLE_ID_2
URLS
    echo "Created sample URL list at: $URL_LIST"
    echo "Please add YouTube URLs to this file, then re-run."
    exit 0
fi

# ---- Build yt-dlp options ----
YT_OPTS=(
    --format "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]"
    --merge-output-format mp4
    --output "${OUTPUT_DIR}/%(title)s_%(id)s.%(ext)s"
    --restrict-filenames
    --no-overwrites
    --rate-limit "$RATE_LIMIT"
    --retries 3
    --fragment-retries 3
    --no-playlist
    --write-info-json
    --no-write-comments
    --quiet
    --progress
)

if [[ -n "$COOKIES" && -f "$COOKIES" ]]; then
    YT_OPTS+=(--cookies "$COOKIES")
fi

# ---- Download ----
echo "============================================"
echo "DIRECTOR Trailer Downloader"
echo "============================================"
echo "URL list:   $URL_LIST"
echo "Output dir: $OUTPUT_DIR"
echo "Rate limit: $RATE_LIMIT"
echo ""

count=0
failed=0
skipped=0

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    line="$(echo "$line" | sed 's/#.*//' | xargs)"
    [[ -z "$line" ]] && continue

    count=$((count + 1))
    if [[ "$MAX_DOWNLOADS" -gt 0 && "$count" -gt "$MAX_DOWNLOADS" ]]; then
        echo "Reached max downloads ($MAX_DOWNLOADS). Stopping."
        break
    fi

    echo "[${count}] Downloading: $line"

    if yt-dlp "${YT_OPTS[@]}" "$line" 2>/dev/null; then
        echo "    OK"
    else
        echo "    FAILED (will continue)"
        failed=$((failed + 1))
    fi

    # Small delay to avoid rate limiting
    sleep 2

done < "$URL_LIST"

echo ""
echo "============================================"
echo "Done. Downloaded: $((count - failed)), Failed: $failed"
echo "Videos saved to: $OUTPUT_DIR"
echo "============================================"

# List downloaded files
echo ""
echo "Downloaded files:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "(no mp4 files found)"
