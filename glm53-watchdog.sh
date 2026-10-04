#!/usr/bin/env bash
# Watchdog du serveur GLM dual Spark, exécuté sur le WORKER (spark-8087) par
# un timer systemd utilisateur (systemd/glm53-watchdog.timer).
#
# Les deux nœuds forment un seul serveur : si l'un redémarre ou plante,
# l'autre garde une session NCCL morte. Le watchdog relance alors les DEUX,
# dans l'ordre de la recette : worker (+ relais) d'abord, head ~25 s après.
# Il couvre aussi le boot : après un redémarrage, rien ne tourne (restart: "no")
# et c'est lui qui remonte l'ensemble.
#
# Pause pour une maintenance manuelle :  touch ~/.local/state/glm53-watchdog/pause
# Reprise :                              rm ~/.local/state/glm53-watchdog/pause
# Journal :                              journalctl --user -u glm53-watchdog
set -uo pipefail

HEAD_SSH=${HEAD_SSH:-ikki@192.168.100.11}
HEAD_DIR=${HEAD_DIR:-infra/vllm-config}          # relatif au home du compte head
HEAD_URL=${HEAD_URL:-http://192.168.100.11:8000}
GRACE_S=${GRACE_S:-1500}                         # 25 min : chargement + compilation
MAX_RESTARTS=${MAX_RESTARTS:-3}                  # au-delà, sur WINDOW_S : on abandonne
WINDOW_S=${WINDOW_S:-21600}                      # 6 h

cd "$(dirname "$(readlink -f "$0")")"
STATE=${XDG_STATE_HOME:-$HOME/.local/state}/glm53-watchdog
mkdir -p "$STATE"
exec 9>"$STATE/lock"
flock -n 9 || exit 0                             # une seule instance à la fois

log() { echo "$*"; }
now=$(date +%s)
ssh_head() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_SSH" "$@"; }

[ -e "$STATE/pause" ] && { log "en pause ($STATE/pause)"; exit 0; }

healthy()      { curl -sf -m 10 "$HEAD_URL/health" >/dev/null; }
worker_up()    { docker ps -q -f name='^glm53-vllm$' | grep -q .; }
head_up()      { ssh_head "docker ps -q -f name='^glm53-vllm$'" | grep -q .; }

if healthy && worker_up && head_up; then
  rm -f "$STATE/failing_since"
  exit 0
fi

# Démarrage en cours : on laisse le temps de charger les poids
last=$(cat "$STATE/last_restart" 2>/dev/null || echo 0)
if (( now - last < GRACE_S )); then
  log "démarrage en cours depuis $(( (now - last) / 60 )) min, on attend"
  exit 0
fi

# Garde-fou contre une boucle de relances sur une panne persistante
recent=$(awk -v t=$((now - WINDOW_S)) '$1 > t' "$STATE/restarts" 2>/dev/null | wc -l)
if (( recent >= MAX_RESTARTS )); then
  log "ABANDON : $recent relances en $((WINDOW_S / 3600)) h sans succès, intervention manuelle requise (puis rm $STATE/restarts)"
  exit 1
fi

if ! ssh_head true 2>/dev/null; then
  log "head injoignable en SSH ($HEAD_SSH) : machine éteinte ou réseau CX7 coupé, on réessaiera"
  exit 0
fi

log "serveur KO (health=$(healthy && echo ok || echo ko) worker=$(worker_up && echo up || echo down) head=$(head_up && echo up || echo down)) : relance des deux nœuds"
echo "$now" >> "$STATE/restarts"
echo "$now" > "$STATE/last_restart"

ssh_head "cd $HEAD_DIR && docker compose down" 2>&1 | tail -1
docker compose down 2>&1 | tail -1
docker compose up -d 2>&1 | tail -2              # worker + relais
sleep 25
ssh_head "cd $HEAD_DIR && docker compose up -d" 2>&1 | tail -1
log "relance lancée ; prochain contrôle de santé dans $((GRACE_S / 60)) min"
