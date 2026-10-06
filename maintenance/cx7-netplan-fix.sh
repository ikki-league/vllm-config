#!/usr/bin/env bash
# [root] Déclare la 2e moitié du port CX7 (enP2p1s0f1np1) dans netplan : sans
# DHCP, MTU 9000. Idempotent, avec retour arrière automatique si le lien CX7
# principal (enp1s0f1np1) perd son adresse après application.
# Programmé par : sudo systemd-run --on-calendar=... /bin/bash <ce script>
set -u
F=/etc/netplan/40-cx7.yaml
LOG=/var/log/cx7-netplan-fix.log
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') $(hostname)"

if grep -q 'enP2p1s0f1np1:' "$F"; then
  echo "bloc déjà présent dans $F"
else
  BAK="$F.bak-$(date +%Y%m%d-%H%M%S)"
  cp -p "$F" "$BAK" && echo "sauvegarde : $BAK"
  printf '    enP2p1s0f1np1:\n      dhcp4: false\n      dhcp6: false\n      mtu: 9000\n' >> "$F"
fi

# Connexions NetworkManager créées d'office sur cette interface : elles
# relanceraient le DHCP en concurrence avec netplan.
nmcli -t -f NAME,DEVICE connection show | awk -F: '$2=="enP2p1s0f1np1" && $1 !~ /^netplan-/ {print $1}' |
  while read -r c; do nmcli connection delete "$c" && echo "connexion NM supprimée : $c"; done

restore() {
  echo "ÉCHEC : $1 — retour arrière"
  [ -n "${BAK:-}" ] && cp -p "$BAK" "$F" && netplan apply && echo "ancienne config réappliquée"
  exit 1
}
netplan generate || restore "netplan generate"
netplan apply || restore "netplan apply"
sleep 15
ip -4 addr show enp1s0f1np1 | grep -q 'inet 192\.168\.100\.' || restore "enp1s0f1np1 sans adresse 192.168.100.x"
echo "mtu enp1s0f1np1=$(cat /sys/class/net/enp1s0f1np1/mtu) enP2p1s0f1np1=$(cat /sys/class/net/enP2p1s0f1np1/mtu)"
nmcli -t -f DEVICE,STATE,CONNECTION device status | grep -E 'enp1s0f1np1|enP2p1s0f1np1'
echo "OK"
