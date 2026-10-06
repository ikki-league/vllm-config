#!/usr/bin/env bash
# [mak, spark-8087] Après la maintenance : vérifie le lien CX7 sur les deux
# nœuds, puis lève la pause du watchdog, qui relance GLM dans l'ordre.
# La pause est levée même si une vérification échoue : le service passe avant.
set -u
S=$HOME/.local/state/glm53-watchdog; LOG=$S/maintenance.log
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') reprise"
for n in local 192.168.100.11; do
  if [ $n = local ]; then r=$(cat /sys/class/net/enP2p1s0f1np1/mtu); else
    r=$(ssh -o BatchMode=yes -o ConnectTimeout=10 mak@$n 'cat /sys/class/net/enP2p1s0f1np1/mtu'); fi
  echo "MTU enP2p1s0f1np1 sur $n : ${r:-?}"
done
ping -c3 -M do -s 8972 -W2 192.168.100.11 >/dev/null && echo "ping jumbo CX7 : OK" || echo "ping jumbo CX7 : ÉCHEC"
rm -f "$S/pause" && echo "pause levée : le watchdog relance GLM au prochain passage (≤ 2 min)"
