# ikki-spark — Qwen3.8-27B-FP8 sur 2x DGX Spark

Ce dépôt sert le modèle **Qwen3.8-27B-FP8** avec vLLM, réparti sur **deux DGX
Spark** reliés en direct par un câble QSFP. Ce guide explique pas à pas
comment passer de « deux machines indépendantes » à « un seul serveur
d'inférence sur deux machines ».

---

## 1. Comprendre l'architecture avant de commencer

```
             clients (API OpenAI)
                     │
                     ▼  http://<IP-LAN-de-Spark-A>:8000
   ┌────────────────────────────┐   câble QSFP 200 Gb/s   ┌────────────────────────────┐
   │ Spark A — HEAD (rank 0)    │◄───────────────────────►│ Spark B — WORKER (rank 1)  │
   │ 192.168.100.10             │   RDMA (RoCE) / NCCL    │ 192.168.100.11             │
   │ • sert l'API sur :8000     │                         │ • pas d'API (--headless)   │
   │ • moitié des poids du      │                         │ • autre moitié des poids   │
   │   modèle + KV cache        │                         │   du modèle + KV cache     │
   └────────────────────────────┘                         └────────────────────────────┘
```

Quelques notions utiles :

- **Tensor parallel (TP=2)** : chaque couche du modèle est coupée en deux.
  Chaque GPU calcule sa moitié, puis les deux machines échangent leurs résultats
  à **chaque couche, pour chaque token**. Le lien entre les deux machines doit
  donc être très rapide et très peu latent : c'est le rôle du câble QSFP
  (ConnectX-7) et de **RDMA**, qui permet aux cartes réseau d'écrire directement
  dans la mémoire de l'autre machine sans passer par le CPU.
- **NCCL** est la bibliothèque NVIDIA qui fait ces échanges. Elle doit utiliser
  le lien CX7 en RDMA (`NET/IB` dans les logs) et pas le Wi-Fi ou le TCP
  (`NET/Socket`), qui serait des dizaines de fois plus lent.
- **Head / worker** : vLLM lance un process par GPU. Le **head** (rank 0)
  coordonne et expose l'API ; le **worker** (rank 1) se contente de calculer.
  On n'envoie jamais de requête au worker.
- **Ce qui est gagné** : la mémoire de deux GB10 au lieu d'une. On a donc
  deux fois plus de place pour le KV cache, d'où `--max-num-seqs 64` au lieu
  de 32.

Les deux machines utilisent **exactement le même dépôt et le même
`docker-compose.yml`**. La seule différence entre elles est leur fichier
`.env`, qui leur dit qui elles sont.

### Conventions de ce guide

| | Spark A | Spark B |
|---|---|---|
| Rôle | head (rank 0) | worker (rank 1) |
| IP sur le lien CX7 | `192.168.100.10` | `192.168.100.11` |
| Expose l'API | oui, port 8000 | non |

Chaque étape indique **où** exécuter les commandes :
🅰️ = sur Spark A, 🅱️ = sur Spark B, 🅰️🅱️ = sur les deux.

---

## 2. Préparer le dépôt 🅰️🅱️

Le service thermique attend le dépôt dans `/home/mak/infra/vllm-config`.
Clonez-le à cet endroit **sur les deux machines** :

```bash
mkdir -p /home/mak/infra
git clone git@github.com:mak-ikki/vllm-config.git /home/mak/infra/vllm-config
cd /home/mak/infra/vllm-config
git checkout qwen3.8-dual-spark
```

> **Pourquoi c'est important :** les arguments passés à vLLM (modèle,
> parallélisme, taille du cache, décodage spéculatif…) doivent être
> **strictement identiques** sur les deux nœuds. S'ils diffèrent, le
> démarrage échoue ou se bloque. Être sur le même commit des deux côtés
> le garantit. Vérifiez avec `git log --oneline -1` sur chaque machine.

---

## 3. Protection thermique 🅰️🅱️

Le problème de surchauffe du GB10 (voir `setup-thermal.sh`) concerne **chaque**
machine. Installez le service sur les deux :

```bash
sudo cp vllm-thermal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-thermal.service
```

**Vérification** : `journalctl -u vllm-thermal.service -b` doit afficher
`Persistence mode enabled` et `GPU clocks capped at 300-2200 MHz`.

---

## 4. Brancher et configurer le lien ConnectX-7

### 4.1 Brancher le câble

Reliez un port QSFP de Spark A à un port QSFP de Spark B (câble direct, sans
switch).

### 4.2 Repérer le port utilisé 🅰️🅱️

```bash
ibdev2netdev
```

Exemple de sortie une fois le câble branché :

```
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)      ◄── port câblé
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)  ◄── même port, 2e moitié
```

> **Pourquoi deux lignes `Up` ?** Sur le DGX Spark, chaque port QSFP physique
> est vu par le système comme **deux interfaces**, une par moitié du bus PCIe.
> Chacune donne environ 100 Gb/s. NCCL utilise les deux en même temps pour
> atteindre 200 Gb/s : c'est pour ça que `CX7_HCA` contient deux noms.

Notez :
- l'**interface** (`enp1s0f1np1` ou `enp1s0f0np0`) → ce sera `CX7_IFNAME` ;
- les **deux HCA** du port (`rocep1s0f1,roceP2p1s0f1` ou
  `rocep1s0f0,roceP2p1s0f0`) → ce sera `CX7_HCA`.

Les valeurs par défaut du dépôt correspondent au cas `enp1s0f1np1`.

### 4.3 Donner une IP fixe à chaque extrémité

Le lien est direct, il n'y a donc pas de DHCP : chaque machine reçoit une IP
fixe dans un petit réseau privé dédié, `192.168.100.0/24`.

🅰️ Sur Spark A :

```bash
sudo cp netplan/40-cx7.yaml /etc/netplan/40-cx7.yaml
# laisser l'adresse 192.168.100.10/24
# si le port câblé est enp1s0f0np0, remplacer le nom d'interface dans le fichier
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

🅱️ Sur Spark B : la même chose, mais en **remplaçant l'adresse** :

```bash
sudo cp netplan/40-cx7.yaml /etc/netplan/40-cx7.yaml
sudo sed -i 's#192.168.100.10/24#192.168.100.11/24#' /etc/netplan/40-cx7.yaml
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

> **Pourquoi MTU 9000 ?** Avec des paquets « jumbo » de 9000 octets au lieu de
> 1500, il faut six fois moins de paquets pour transférer la même quantité de
> données. Le MTU doit être **le même des deux côtés**, sinon les gros paquets
> sont perdus.

### 4.4 Tester le lien

🅰️ `./check-cx7.sh 192.168.100.11`
🅱️ `./check-cx7.sh 192.168.100.10`

Résultat attendu sur chaque machine :

```
[OK] enp1s0f1np1 UP, MTU 9000
[OK] 192.168.100.x joignable en MTU 9000
[OK] Poids Qwen3.8-27B-FP8 présents
```

Si le port câblé n'est pas `enp1s0f1np1`, passez-le en 2e argument :
`./check-cx7.sh 192.168.100.11 enp1s0f0np0`.

### 4.5 Ouvrir le pare-feu sur le lien CX7 🅰️🅱️

Si UFW est actif (politique `DROP` par défaut), le worker ne peut pas joindre
le rendez-vous du head (`29501`) ni les ports aléatoires qu'ouvrent NCCL et
vLLM entre les nœuds : le head attend alors 10 minutes puis échoue avec
`Timed out ... waiting for clients. 1/2 clients joined.` Le lien étant direct
et privé, on autorise tout le trafic qui arrive par lui :

```bash
sudo ufw allow in on enp1s0f1np1 from 192.168.100.0/24
sudo ufw status verbose
```

(adapter le nom d'interface si le port câblé est `enp1s0f0np0`). Les
transferts RDMA eux-mêmes ne passent pas par netfilter, mais le rendez-vous
et le bootstrap NCCL sont en TCP.

---

## 5. Mettre les poids du modèle sur les deux machines 🅰️🅱️

Chaque nœud charge **sa moitié** du modèle depuis son disque local
(`/home/mak/ai/models`). Les poids doivent donc être présents **sur les deux
machines**.

Si Spark A les a déjà, le plus rapide est de les copier vers Spark B par le
lien CX7 :

```bash
# 🅰️ depuis Spark A
rsync -a --info=progress2 /home/mak/ai/models/ mak@192.168.100.11:/home/mak/ai/models/
```

Sinon, chaque machine les téléchargera elle-même au premier démarrage, ce qui
est plus long (et nécessite `HUGGING_FACE_HUB_TOKEN` dans le `.env`).

---

## 6. Construire l'image 🅰️🅱️

```bash
cd /home/mak/infra/vllm-config
docker compose build
```

L'image doit être construite **sur chaque machine** : elle est locale à chaque
démon Docker.

---

## 7. Créer le fichier `.env` de chaque machine

C'est **la seule différence** entre les deux machines. Partez du modèle :

```bash
cp .env.example .env
```

🅰️ Sur Spark A, le bloc head, actif par défaut, convient tel quel :

```ini
NODE_RANK=0
NODE_IP=192.168.100.10
VLLM_ROLE_ARGS=--host 0.0.0.0 --port 8000
```

🅱️ Sur Spark B, commentez le bloc head et décommentez le bloc worker :

```ini
NODE_RANK=1
NODE_IP=192.168.100.11
VLLM_ROLE_ARGS=--headless
```

Sur **les deux**, vérifiez la partie commune :

```ini
HUGGING_FACE_HUB_TOKEN=hf_...        # si les poids doivent être téléchargés
MASTER_ADDR=192.168.100.10           # toujours l'IP de Spark A, même sur Spark B
MASTER_PORT=29501
CX7_IFNAME=enp1s0f1np1               # relevé à l'étape 4.2
CX7_HCA=rocep1s0f1,roceP2p1s0f1      # relevé à l'étape 4.2
NCCL_DEBUG=INFO                      # INFO pour le premier démarrage, WARN ensuite
```

Le rôle de chaque variable :

| Variable | Rôle |
|---|---|
| `NODE_RANK` | numéro du nœud : 0 = head, 1 = worker |
| `NODE_IP` | IP **de cette machine** sur le lien CX7 ; vLLM l'annonce à l'autre nœud |
| `VLLM_ROLE_ARGS` | head : ouvre l'API ; worker : `--headless`, pas d'API |
| `MASTER_ADDR` / `MASTER_PORT` | point de rendez-vous : le worker se connecte au head à cette adresse |
| `CX7_IFNAME` | interface par laquelle NCCL fait le rendez-vous et les échanges TCP |
| `CX7_HCA` | cartes RDMA que NCCL utilise pour les vrais transferts de données |

> Si une variable obligatoire manque, `docker compose` refuse de démarrer avec
> un message explicite (`NODE_RANK manquant dans .env`, par exemple).

**Vérification** : `docker compose config | grep -A1 -E "node-rank|headless|--host"`
doit montrer `--node-rank "0"` et `--host` sur A, et `--node-rank "1"` et
`--headless` sur B.

---

## 8. Démarrer

Lancez de préférence le head en premier, puis le worker dans la foulée :

```bash
# 🅰️
docker compose up -d
# 🅱️
docker compose up -d
```

Suivez les logs **sur les deux machines** :

```bash
docker compose logs -f
```

Ce qui se passe pendant le démarrage, qui prend plusieurs minutes :

1. Le head ouvre le point de rendez-vous sur `192.168.100.10:29501` et
   **attend** le worker. Il est normal qu'il semble bloqué tant que B n'est
   pas lancé.
2. Le worker s'y connecte, et NCCL initialise le lien RDMA. Avec
   `NCCL_DEBUG=INFO`, on doit voir des lignes contenant **`NET/IB`** sur les
   deux machines. Si on voit `NET/Socket`, voir le dépannage ci-dessous.
3. Chaque nœud charge sa moitié des poids, puis capture les CUDA graphs.
4. Le head affiche `Application startup complete` et répond sur `:8000`.

---

## 9. Vérifier que tout fonctionne 🅰️

```bash
curl -s http://localhost:8000/v1/models | jq
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B-FP8","messages":[{"role":"user","content":"Bonjour !"}]}' | jq
```

Pendant une génération, `nvidia-smi` doit montrer les GPU **des deux
machines** occupés.

Une fois que tout marche, repassez `NCCL_DEBUG=WARN` dans les deux `.env`
pour alléger les logs.

---

## 10. Exploitation au quotidien

| Action | Commande |
|---|---|
| Arrêter | 🅰️🅱️ `docker compose down` |
| Redémarrer | 🅰️🅱️ `docker compose restart` (sur **les deux**, voir ci-dessous) |
| Mettre à jour la config | 🅰️🅱️ `git pull && docker compose build && docker compose up -d` |
| Logs | 🅰️🅱️ `docker compose logs -f` |

> ⚠️ **Les deux nœuds forment un seul serveur.** Si l'un redémarre ou plante,
> l'autre perd la communication NCCL et ne peut pas la rétablir seul. Il faut
> **toujours redémarrer les deux**. De même, une modification de
> `docker-compose.yml` doit être appliquée des deux côtés.

### Accès depuis d'autres conteneurs

Le conteneur utilise le réseau de l'hôte (`network_mode: host`), nécessaire
pour RDMA. L'ancien réseau Docker `hermes-net` n'existe donc plus. Un client
qui tourne lui-même dans un conteneur doit joindre l'API par l'IP de Spark A
sur le LAN, ou par `host.docker.internal:8000` s'il tourne sur Spark A et
déclare :

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

---

## 11. Dépannage

| Symptôme | Cause probable | À vérifier |
|---|---|---|
| Le head reste bloqué au démarrage | le worker n'est pas lancé ou ne joint pas le head | `docker compose ps` sur B ; `MASTER_ADDR` identique des deux côtés ; `ping 192.168.100.10` depuis B ; pare-feu ouvert sur le lien CX7 (étape 4.5) ; erreur `1/2 clients joined` après 10 min |
| `NET/Socket` au lieu de `NET/IB` dans les logs | NCCL ne trouve pas les cartes RDMA | `CX7_HCA` correspond au port câblé (`ibdev2netdev`) ; `/dev/infiniband` existe sur l'hôte |
| Le ping jumbo de `check-cx7.sh` échoue | MTU différent des deux côtés | `ip link show <iface>` → `mtu 9000` sur A et B |
| Erreur de forme ou de config au chargement | les deux nœuds n'ont pas les mêmes arguments | `git log --oneline -1` identique sur A et B ; `docker compose build` refait des deux côtés |
| Timeout NCCL après un redémarrage d'un seul nœud | l'autre nœud tient une session NCCL morte | redémarrer **les deux** : `docker compose restart` sur A et B |
| Réponses vides ou remplies de `!!!!` | bug amont prefix caching + MTP sur modèle hybride (vllm#53912) | retirer `--enable-prefix-caching` du compose sur A et B |
| Un nœud télécharge le modèle au démarrage | poids absents localement | étape 5 |
| Coupure ou ralentissement sous charge | thermique | étape 3, `systemctl status vllm-thermal` sur **les deux** machines |

---

## Fichiers du dépôt

| Fichier | Rôle |
|---|---|
| `docker-compose.yml` | service vLLM, identique sur les deux nœuds |
| `.env.example` | modèle de `.env` (rôle du nœud, IP, interfaces) |
| `Dockerfile` | image vLLM + transformers récent + chat template |
| `unsloth.jinja` | chat template Qwen3.8 (raisonnement `<think>` et appels d'outils) |
| `netplan/40-cx7.yaml` | IP fixe du lien direct ConnectX-7 |
| `check-cx7.sh` | vérification du lien et des poids avant démarrage |
| `setup-thermal.sh`, `vllm-thermal.service` | limitation des fréquences GPU (GB10) |
