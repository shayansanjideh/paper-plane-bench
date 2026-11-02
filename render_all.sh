#!/bin/bash
cd "$(dirname "$0")"
render() { python3 video.py evolution results/$1/$2.json videos/$1.mp4 "$3" && echo "done $1"; }
render opus run0 "Claude Opus 5.5" &
render sonnet run0 "Claude Sonnet 5.5" &
render gpt-6.1-sol run2 "GPT-6.1 Sol" &
render haiku run2 "Claude Haiku 5.5" &
wait
render gpt-6.1-terra run0 "GPT-6.1 Terra" &
render gpt-6-luna run2 "GPT-6 Luna" &
render cma run0 "CMA-ES optimizer (no LLM)" &
render opus_blind run2 "Opus, blind (never sees results)" &
wait
echo ALLDONE
