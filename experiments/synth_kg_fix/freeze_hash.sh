#!/bin/sh
# usage: freeze_hash.sh  -> prints two hashes
cd "$(dirname "$0")/../.."
A=$(git diff -- src tests configs prompts | shasum -a 256 | cut -d' ' -f1)
B=$(find src tests configs prompts -type f -not -name '*.pyc' -not -path '*/__pycache__/*' | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | cut -d' ' -f1)
echo "git_diff_sha256=$A"
echo "all_files_sha256=$B"
