#!/usr/bin/env bash
# [mak, spark-8087] Bilan ~30 min après la reprise.
S=$HOME/.local/state/glm53-watchdog; LOG=$S/maintenance.log
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') bilan"
echo "health relais : $(curl -s -m10 -o /dev/null -w '%{http_code}' http://localhost:8000/health)"
echo "avertissements GID depuis la reprise (worker) : $(docker logs --since 25m glm53-vllm 2>&1 | grep -c 'GID table changed')"
