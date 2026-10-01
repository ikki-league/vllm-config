# ikki-spark — Qwen3.8-27B-FP8 sur 2x DGX Spark

Le modèle est servi en tensor parallel (TP=2) sur deux DGX Spark reliés en
direct par un câble QSFP (ConnectX-7, 200 Gb/s). On utilise le backend
multi-nœud natif de vLLM (`--distributed-executor-backend mp`, sans Ray) :
le nœud 0 (head) sert l'API OpenAI sur `:8000`, le nœud 1 tourne en `--headless`.

## Mise en place (une fois, sur chaque nœud)

1. **Thermique** : installer `vllm-thermal.service` comme avant (sur les **deux** nœuds).
2. **Lien CX7** : copier `netplan/40-cx7.yaml` dans `/etc/netplan/`, adapter
   l'adresse (head `192.168.100.10`, worker `192.168.100.11`), puis
   `sudo chmod 600 /etc/netplan/40-cx7.yaml && sudo netplan apply`.
3. **Poids** : `Qwen/Qwen3.8-27B-FP8` doit être présent dans `/home/mak/ai/models`
   sur chaque nœud (sinon chacun le télécharge au démarrage).
4. **Dépôt** : le même commit des deux côtés (les arguments vLLM doivent être
   strictement identiques), puis `docker compose build`.
5. **.env** : `cp .env.example .env` et choisir le bloc head ou worker.
6. **Vérif** : `./check-cx7.sh <ip-de-l-autre-noeud>`.

## Démarrage

```bash
docker compose up -d        # sur les deux nœuds, dans n'importe quel ordre
docker compose logs -f      # le head attend le worker avant de charger le modèle
```

Pour vérifier que NCCL passe bien par RDMA : `NCCL_DEBUG=INFO` dans `.env`, puis
chercher `NET/IB` dans les logs (`NET/Socket` = repli TCP, beaucoup plus lent).

## Points d'attention

- Le conteneur est en `network_mode: host` (nécessaire pour NCCL/RoCE) : le
  réseau bridge `hermes-net` n'existe plus. Les clients en conteneur doivent
  joindre l'API via l'IP de l'hôte head (ou `host.docker.internal:8000`
  avec `extra_hosts: ["host.docker.internal:host-gateway"]`).
- Si un des deux conteneurs redémarre, l'autre perd le groupe NCCL : relancer
  les deux (`docker compose restart` sur chaque nœud).
- `--max-num-seqs` passe de 32 à 64 : le KV cache disponible double avec TP=2.
