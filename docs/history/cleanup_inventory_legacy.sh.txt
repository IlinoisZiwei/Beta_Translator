#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash cleanup.sh [--dry-run | --help]' \
        'Read-only inventory of paths considered by the former reset script.' \
        'No files are deleted or changed; there is no deletion mode.'
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi

case "${1:-}" in
    ''|--dry-run) ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unsupported argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

printf '%s\n' \
    'Dry run: no files will be deleted or changed.' \
    'These paths were targeted by the old reset script; review them individually.' \
    'Disk usage (KiB), including directory contents:'

for path in models/pose_landmarker.task outputs .setup_done .venv; do
    if [[ -e "$path" || -L "$path" ]]; then
        du -sk -- "$path"
    else
        printf 'missing\t%s\n' "$path"
    fi
done

printf '\n%s\n' \
    'Retain outputs/: videos, reviewed maps, annotations, caches, and diagnostics.' \
    'Models and the working environment may be needed for future runs.' \
    'Review and back up specific files before any separately approved deletion.'
