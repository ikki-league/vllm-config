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

# UFW actif bloque le rendez-vous vLLM et le bootstrap NCCL (voir README 4.5)
if grep -q '^ENABLED=yes' /etc/ufw/ufw.conf 2>/dev/null; then
  echo "[WARN] UFW actif : vérifier 'sudo ufw status' -> ALLOW IN on $IFNAME from 192.168.100.0/24"
fi

# Les poids doivent être présents localement sur chaque nœud
ls -d /home/mak/ai/models/GLM-5.3-Flash-NVFP4/model.safetensors.index.json >/dev/null 2>&1 \
  && echo "[OK] Poids GLM-5.3-Flash-NVFP4 présents" \
  || echo "[KO] Poids GLM absents : lancer ./prepare-model.sh (README étape 5)"
ls /home/mak/ai/glm53/patches/kv_cache_coordinator.py /home/mak/ai/glm53/patches/sparse_attn_indexer_kpool.py >/dev/null 2>&1 \
  && echo "[OK] Correctifs vLLM présents" \
  || echo "[KO] Correctifs absents : lancer ./prepare-model.sh"
