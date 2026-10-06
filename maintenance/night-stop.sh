#!/usr/bin/env bash
# [mak, spark-8087] Avant la maintenance : met le watchdog en pause, arrête
# proprement le head (ikki@promaxgb10) puis le worker et le relais.
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
S=$HOME/.local/state/glm53-watchdog; LOG=$S/maintenance.log; mkdir -p "$S"
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') arrêt pour maintenance"
touch "$S/pause"
ssh -o BatchMode=yes -o ConnectTimeout=10 ikki@192.168.100.11 'cd infra/vllm-config && docker compose down' 2>&1 | tail -1
docker compose down 2>&1 | tail -1
