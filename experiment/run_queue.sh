#!/bin/sh
# Q-Pain on the four main models first (both cue types, 3 seeds), then the
# scale-and-reasoning stage, which resumes from the responses already on disk.
cd "$(dirname "$0")"
export OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}   # set to the study host
for m in gemma3:12b qwen3:14b ministral-3:14b llama3.2:latest; do
  echo "$(date '+%F %T') qpain start $m"; python3.10 qpain.py --models "$m" --seeds 3; echo "$(date '+%F %T') qpain done  $m"
done
echo "$(date '+%F %T') qpain complete; resuming the scale stage"
sh run_stage.sh
