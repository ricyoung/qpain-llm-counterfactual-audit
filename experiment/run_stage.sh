#!/bin/sh
# Scale-and-reasoning extension: larger open-weight models, thinking on/off and
# effort levels, on the v2 vignettes with explicit labels and name cues.
# Resumable: saved responses are skipped, so re-running continues where it stopped.
cd "$(dirname "$0")"
export OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}   # set to the study host
run() { echo "$(date '+%F %T') start $*"; python3.10 run.py --vignettes vignettes_v2.json --reps 5 "$@"; echo "$(date '+%F %T') done  $*"; }
for spec in "qwen3.8:27b off" "qwen3.8:27b on" "gpt-oss:20b low" "gpt-oss:20b high" "qwen3.6:35b off" "qwen3.6:35b on" "qwen3:14b on"; do
  set -- $spec
  run --models "$1" --think "$2"
  run --models "$1" --think "$2" --names
done
echo "$(date '+%F %T') stage complete"
