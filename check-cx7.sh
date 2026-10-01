#!/usr/bin/env bash
# Vérifie le lien ConnectX-7 vers l'autre DGX Spark avant de lancer vLLM.
# Usage: ./check-cx7.sh <ip-du-pair> [interface]
set -euo pipefail

PEER="${1:?usage: $0 <ip-du-pair> [interface]}"
IFNAME="${2:-enp1s0f1np1}"

echo "=== Lien CX7 ($IFNAME) ==="
ip -br addr show "$IFNAME"
ip link show "$IFNAME" | grep -q "state UP" || { echo "[KO] $IFNAME n'est pas UP (câble QSFP ?)"; exit 1; }
echo "[OK] $IFNAME UP, MTU $(cat /sys/class/net/$IFNAME/mtu)"

ibdev2netdev | grep -F "$IFNAME"

# Ping avec paquet jumbo non fragmenté : valide aussi le MTU 9000 des deux côtés
ping -c 3 -M do -s 8972 "$PEER" >/dev/null && echo "[OK] $PEER joignable en MTU 9000" \
  || { echo "[KO] $PEER injoignable en MTU 9000"; exit 1; }

# Les poids doivent être présents localement sur chaque nœud
ls -d /home/mak/ai/models/hub/models--Qwen--Qwen3.8-27B-FP8 >/dev/null 2>&1 \
  && echo "[OK] Poids Qwen3.8-27B-FP8 présents" \
  || echo "[WARN] Poids absents de /home/mak/ai/models : ils seront téléchargés au démarrage"
