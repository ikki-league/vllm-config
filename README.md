# ikki-spark — GLM-5.3 Flash NVFP4 sur 2x DGX Spark

Cette branche sert **GLM-5.3 Flash** (MoE 320B, 18B actifs, poids NVFP4
NVIDIA de ~205 Go) avec vLLM, réparti sur **deux DGX
Spark** reliés en direct par un câble QSFP. Ce guide explique pas à pas
comment passer de « deux machines indépendantes » à « un seul serveur
d'inférence sur deux machines ».

> Recette [tonyd2wild CURRENT.md](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/blob/d061f26ad3ec5c3c04f64aad7516dbe62a8aa6af/CURRENT.md)
> (@d061f26) : image vLLM patchée sm121, décodage spéculatif DFlash2, 262K de
> contexte, 6 requêtes simultanées. Mesuré par l'auteur : ~45 tok/s pour une
> requête seule (19 en prose, 52 en code), ~85 tok/s cumulés à 6 requêtes.
> Le modèle **ne tient pas sur une seule Spark** : TP=2 est obligatoire. Le
> drafter DFlash2 est sous licence **CC-BY-NC-ND 4.0** (non commercial). Pour revenir à Qwen3.8 : `git checkout qwen3.8-dual-spark`
> sur les deux nœuds.

---

## 1. Comprendre l'architecture avant de commencer

```
             clients (API OpenAI)
                     │
                     ▼  http://<IP-LAN-de-Spark-A>:8000
   ┌────────────────────────────┐   câble QSFP 200 Gb/s   ┌────────────────────────────┐
   │ Spark A — HEAD (rank 0)    │◄───────────────────────►│ Spark B — WORKER (rank 1)  │
   │ promaxgb10-e6a6, ikki      │   RDMA (RoCE) / NCCL    │ spark-8087, mak            │
   │ 192.168.100.11             │                         │ 192.168.100.10             │
   │ • sert l'API sur :8000     │                         │ • pas d'API (--headless)   │
   │                            │                         │ • relais :8000 → A pour    │
   │                            │                         │   les clients locaux       │
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
- **Ce qui est gagné** : la mémoire de deux GB10 au lieu d'une, soit
  ~242 GiB. Les poids en prennent ~87 GiB par nœud : il reste peu de place,
  d'où un KV cache fp8 épinglé à 6 GiB (560K tokens), au plus 6 requêtes
  simultanées et pas de CUDA graphs (`--enforce-eager`).

### Le modèle et ses réglages, en clair

- **MoE (Mixture of Experts)** : GLM-5.3 Flash a 320 milliards de paramètres,
  mais pour chaque token il n'en active que 18 milliards (quelques « experts »
  choisis parmi des centaines). On paie la **mémoire** d'un modèle de 320B,
  mais la **vitesse** ressemble à celle d'un modèle de 18B : c'est ce qui le
  rend utilisable sur des Spark, dont la mémoire est grande mais lente
  (~273 Go/s).
- **NVFP4** : les poids des experts sont stockés sur 4 bits au lieu de 16, ce
  qui divise leur taille par ~3,5. Le GB10 (architecture Blackwell) sait
  calculer directement dans ce format. L'**attention** reste en haute
  précision dans le checkpoint NVIDIA : c'est pour ça qu'on n'utilise que
  celui-là (voir étape 5).
- **Décodage spéculatif (DFlash2)** : générer un token oblige à relire tous
  les poids actifs en mémoire, ce qui est lent. Un petit modèle annexe, le
  **drafter** (2,2 Go), *devine* 7 tokens d'avance ; le gros modèle les
  **vérifie tous en une seule passe** et garde ceux qui sont justes. Le texte
  produit est identique à celui du modèle seul, mais 2 à 3 fois plus vite
  quand le drafter devine bien (code, JSON, appels d'outils) et moins sur de
  la prose libre.
- **KV cache** : la « mémoire de travail » des conversations en cours. Chaque
  token déjà lu y laisse une trace pour ne pas être recalculé. Il est stocké
  en fp8 et **épinglé à 6 GiB** par nœud (560 000 tokens, soit deux
  conversations de 262K) : on lui fixe une taille au lieu de laisser vLLM la
  deviner, parce que cette estimation est peu fiable sur la mémoire unifiée
  des Spark et finit en plantage.
- **Prefix cache** : si une requête commence comme une précédente (cas typique
  d'un agent, qui renvoie toute la conversation à chaque tour), le début est
  repris du KV cache au lieu d'être recalculé.
- **Image patchée** : le support de GLM-5.3 sur GB10 (sm_121) est récent. Le
  vLLM officiel plante de plusieurs façons sur ce matériel ; l'image de
  tonyd2wild corrige 7 bugs, et deux correctifs plus récents sont montés
  par-dessus au démarrage (étape 5).
- **Thinking** : le modèle « réfléchit » toujours avant de répondre, plus ou
  moins longuement selon le niveau d'effort (`low`, `high`, `max`). Le
  serveur règle `low` par défaut ; un client peut demander plus par requête.

Budget mémoire d'un nœud (~121 GiB utilisables) :

| Poste | Taille |
|---|---|
| moitié des poids du modèle | ~87 GiB |
| drafter DFlash2 | ~1,3 GiB |
| KV cache épinglé | 6 GiB |
| activations, buffers, système, autres services | le reste, quelques GiB |

Il ne reste que 1 à 2 GiB libres sous charge : **rien d'autre de lourd ne
doit tourner sur les Spark** (ni Qwen, ni un autre modèle).

Les deux machines utilisent **exactement le même dépôt et le même
`docker-compose.yml`**. La seule différence entre elles est leur fichier
`.env`, qui leur dit qui elles sont.

### Conventions de ce guide

| | Spark A | Spark B |
|---|---|---|
| Machine | `promaxgb10-e6a6` | `spark-8087` |
| Compte qui lance Docker | `ikki` | `mak` |
| Rôle | head (rank 0) | worker (rank 1) |
| IP sur le lien CX7 | `192.168.100.11` | `192.168.100.10` |
| Dépôt | `/home/ikki/infra/vllm-config` | `/home/mak/infra/vllm-config` |
| Données (`AI_DIR`) | `/home/ikki/ai` | `/home/mak/ai` |
| Expose l'API | oui, port 8000 | relais : son `:8000` renvoie vers A |

> **Pourquoi un relais sur B ?** Les clients (Hermes, nanoclaw, hybrid-llm,
> devcontainers ikki via WireGuard `10.200.0.1`) tournent sur `spark-8087`
> et appellent `:8000` en local. Plutôt que de les reconfigurer un par un —
> et les pairs WireGuard ne voient de toute façon pas le lien CX7 —, un petit
> conteneur `socat` (`glm53-relay`, activé par `COMPOSE_PROFILES=relay`)
> écoute sur le `:8000` de B et transmet au head par le câble CX7.

Chaque étape indique **où** exécuter les commandes :
🅰️ = sur Spark A, 🅱️ = sur Spark B, 🅰️🅱️ = sur les deux.

---

## 2. Préparer le dépôt 🅰️🅱️

Clonez le dépôt dans le home du compte qui lance Docker sur chaque machine
(`ikki` sur A, `mak` sur B) :

```bash
mkdir -p ~/infra
git clone git@github.com:mak-ikki/vllm-config.git ~/infra/vllm-config
cd ~/infra/vllm-config
git checkout glm5.3-flash-dual-spark
```

> Le service thermique (étape 3) pointe vers `/home/mak/infra/vllm-config` :
> sur A, adaptez le chemin de `ExecStart` dans `vllm-thermal.service` si ce
> clone-là n'existe pas.

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
# laisser l'adresse 192.168.100.11/24 (celle du head)
# si le port câblé est enp1s0f0np0, remplacer le nom d'interface dans le fichier
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

🅱️ Sur Spark B : la même chose, mais en **remplaçant l'adresse** :

```bash
sudo cp netplan/40-cx7.yaml /etc/netplan/40-cx7.yaml
sudo sed -i 's#192.168.100.11/24#192.168.100.10/24#' /etc/netplan/40-cx7.yaml
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

> **Pourquoi MTU 9000 ?** Avec des paquets « jumbo » de 9000 octets au lieu de
> 1500, il faut six fois moins de paquets pour transférer la même quantité de
> données. Le MTU doit être **le même des deux côtés**, sinon les gros paquets
> sont perdus.

### 4.4 Tester le lien

🅰️ `./check-cx7.sh 192.168.100.10`
🅱️ `./check-cx7.sh 192.168.100.11`

Résultat attendu sur chaque machine :

```
[OK] enp1s0f1np1 UP, MTU 9000
[OK] 192.168.100.x joignable en MTU 9000
[OK] Poids GLM-5.3-Flash-NVFP4 présents
```

Si le port câblé n'est pas `enp1s0f1np1`, passez-le en 2e argument :
`./check-cx7.sh 192.168.100.10 enp1s0f0np0`.

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

## 5. Préparer les poids, l'image et les correctifs 🅰️🅱️

Chaque nœud charge **sa moitié** du modèle depuis son disque local. Tout ce
que le compose monte depuis l'hôte est préparé par un seul script, à
révisions figées :

| Quoi | Où | Taille |
|---|---|---|
| poids `nvidia/GLM-5.3-Flash-NVFP4` | `$AI_DIR/models/GLM-5.3-Flash-NVFP4` | ~205 Go |
| drafter `incoai/GLM-5.3-Flash-DFlash2` | `$AI_DIR/models/GLM-5.3-Flash-DFlash2` | 2,2 Go |
| image `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` | Docker local | |
| correctif top-k SM121 et correctif prefix cache du drafter | `$AI_DIR/glm53/patches` | |

> **Pourquoi ces deux correctifs ?** L'image publiée date du 28/08. Sans le
> premier, le moteur meurt dès qu'un contexte dépasse ~24K tokens. Sans le
> second, le prefix cache ne sert à rien avec DFlash2 : chaque tour d'un agent
> re-calcule toute la conversation (13K tokens : 21 s au lieu de 6 s).

> **Temps de téléchargement** : ~220 Go au total (poids + image de 13 Gio
> compressée). En Wi-Fi (~4 Mo/s mesurés), comptez **plus de 15 heures** ;
> branchez le port RJ45 10 GbE (`enP7s7`) de Spark A pour tomber à moins
> d'une heure si votre accès Internet suit. Spark B n'a rien à télécharger
> d'Internet pour les poids : il les reçoit de A par le lien CX7.

Sur Spark A, lancez le script complet :

```bash
# 🅰️
./prepare-model.sh
```

Copiez ensuite les poids vers Spark B par le lien CX7, bien plus rapide qu'un
second téléchargement :

```bash
# 🅰️
rsync -a --info=progress2 ~/ai/models/GLM-5.3-Flash-NVFP4 \
  ~/ai/models/GLM-5.3-Flash-DFlash2 mak@192.168.100.10:/home/mak/ai/models/
```

puis lancez le même script sur Spark B : il vérifie les poids copiés au lieu
de les retélécharger, et prépare l'image et les correctifs.

```bash
# 🅱️
./prepare-model.sh
```

> N'utilisez pas d'autres checkpoints NVFP4 (LibertAI, versions
> « abliterated ») : ceux qui quantifient l'attention corrompent
> silencieusement des tokens dans les appels d'outils
> ([vllm#54150](https://github.com/vllm-project/vllm/issues/54150)).

---

## 7. Créer le fichier `.env` de chaque machine

C'est **la seule différence** entre les deux machines. Partez du modèle :

```bash
cp .env.example .env
```

🅰️ Sur Spark A, le bloc head, actif par défaut, convient tel quel :

```ini
NODE_RANK=0
NODE_IP=192.168.100.11
VLLM_ROLE_ARGS=--host 0.0.0.0 --port 8000
AI_DIR=/home/ikki/ai
```

🅱️ Sur Spark B, commentez le bloc head et décommentez le bloc worker :

```ini
NODE_RANK=1
NODE_IP=192.168.100.10
VLLM_ROLE_ARGS=--headless
AI_DIR=/home/mak/ai
COMPOSE_PROFILES=relay               # lance aussi le relais :8000 -> A
```

Sur **les deux**, vérifiez la partie commune :

```ini
HUGGING_FACE_HUB_TOKEN=hf_...        # si les poids doivent être téléchargés
MASTER_ADDR=192.168.100.11           # toujours l'IP de Spark A, même sur Spark B
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
| `AI_DIR` | dossier des poids, correctifs et caches de **cette machine** (home du compte qui lance Docker) |
| `COMPOSE_PROFILES` | `relay` sur le worker seulement : ajoute le relais `:8000` vers le head |
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

Le modèle occupe presque toute la mémoire unifiée. **Avant chaque
démarrage**, sur les deux machines :

1. arrêtez tout autre modèle (`docker ps` : plus de `qwen38-vllm`, rien
   d'autre de gros sur le GPU). Pour basculer depuis Qwen3.8, arrêtez-le
   **avant** de changer de branche, tant que le compose Qwen est encore en
   place :

   ```bash
   # 🅰️🅱️
   cd ~/infra/vllm-config
   docker compose down                      # arrête qwen38-vllm
   git fetch && git checkout glm5.3-flash-dual-spark
   ```

   Retour arrière, même principe dans l'autre sens : `docker compose down`,
   `git checkout qwen3.8-dual-spark`, `docker compose build`, cache vidé, puis
   `docker compose up -d` sur les deux nœuds ;
2. videz le page cache, que vLLM compte à tort comme occupé sur la mémoire
   unifiée :

```bash
# 🅰️🅱️
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

Lancez le **worker en premier**, puis le head environ 25 secondes après :

```bash
# 🅱️
docker compose up -d
# 🅰️ (~25 s plus tard)
docker compose up -d
```

Suivez les logs **sur les deux machines** :

```bash
docker compose logs -f
```

Ce qui se passe pendant le démarrage, qui prend environ 15 minutes :

1. Le head ouvre le point de rendez-vous sur `192.168.100.11:29501` et
   **attend** le worker. Il est normal qu'il semble bloqué tant que B n'est
   pas lancé.
2. Le worker s'y connecte, et NCCL initialise le lien RDMA. Avec
   `NCCL_DEBUG=INFO`, on doit voir des lignes contenant **`NET/IB`** sur les
   deux machines. Si on voit `NET/Socket`, voir le dépannage ci-dessous.
3. Chaque nœud charge sa moitié des poids (~100 Go).
4. Le head affiche `Application startup complete` et répond sur `:8000`.

---

## 9. Vérifier que tout fonctionne 🅰️

```bash
curl -s http://localhost:8000/v1/models | jq
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Bonjour !"}]}' | jq
```

Pendant une génération, `nvidia-smi` doit montrer les GPU **des deux
machines** occupés.

Les clients doivent utiliser le nom de modèle `glm-5.3-flash` :
ceux configurés pour `Qwen/Qwen3.8-27B-FP8` recevront une erreur 404.

Le modèle **réfléchit toujours** avant de répondre : son chat template n'a
pas d'interrupteur, seulement un niveau d'effort, `low`, `high` ou `max`.
Le serveur impose `low` par défaut (réponses et appels d'outils rapides) ; un
client monte le niveau par requête avec
`"chat_template_kwargs": {"reasoning_effort": "high"}`.

> ⚠️ Toute autre valeur, par exemple `medium` envoyé par certains clients,
> retombe sur **`max`**, le niveau le plus long : configurez les clients en
> `low` ou `high`.

Une fois que tout marche, repassez `NCCL_DEBUG=WARN` dans les deux `.env`
pour alléger les logs.

---

## 10. Exploitation au quotidien

| Action | Commande |
|---|---|
| Arrêter | 🅰️🅱️ `docker compose down` |
| Redémarrer | 🅰️🅱️ `docker compose restart` (sur **les deux**, voir ci-dessous) |
| Mettre à jour la config | 🅰️🅱️ `git pull && ./prepare-model.sh && docker compose up -d` |
| Logs | 🅰️🅱️ `docker compose logs -f` |

> ⚠️ **Les deux nœuds forment un seul serveur.** Si l'un redémarre ou plante,
> l'autre perd la communication NCCL et ne peut pas la rétablir seul. Il faut
> **toujours redémarrer les deux**. De même, une modification de
> `docker-compose.yml` doit être appliquée des deux côtés.

### Accès depuis d'autres conteneurs

Le conteneur utilise le réseau de l'hôte (`network_mode: host`), nécessaire
pour RDMA. Un client joint l'API :

- sur B, par `localhost:8000`, `172.17.0.1:8000` ou `10.200.0.1:8000`
  (WireGuard) : c'est le relais qui répond et transmet à A ;
- ailleurs, par `192.168.100.11:8000` (lien CX7) ou l'IP LAN de A.

Un client qui tourne lui-même dans un conteneur sur B peut utiliser
`host.docker.internal:8000` s'il déclare :

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

---

## 11. Dépannage

| Symptôme | Cause probable | À vérifier |
|---|---|---|
| Le head reste bloqué au démarrage | le worker n'est pas lancé ou ne joint pas le head | `docker compose ps` sur B ; `MASTER_ADDR` identique des deux côtés ; `ping 192.168.100.11` depuis B ; pare-feu ouvert sur le lien CX7 (étape 4.5) ; erreur `1/2 clients joined` après 10 min |
| `NET/Socket` au lieu de `NET/IB` dans les logs | NCCL ne trouve pas les cartes RDMA | `CX7_HCA` correspond au port câblé (`ibdev2netdev`) ; `/dev/infiniband` existe sur l'hôte |
| Le ping jumbo de `check-cx7.sh` échoue | MTU différent des deux côtés | `ip link show <iface>` → `mtu 9000` sur A et B |
| Erreur de forme ou de config au chargement | les deux nœuds n'ont pas les mêmes arguments | `git log --oneline -1` identique sur A et B ; `./prepare-model.sh` relancé des deux côtés |
| Timeout NCCL après un redémarrage d'un seul nœud | l'autre nœud tient une session NCCL morte | redémarrer **les deux** : `docker compose restart` sur A et B |
| Le moteur meurt sur une conversation de plus de ~24K tokens | correctif top-k absent | `ls $AI_DIR/glm53/patches` sur A et B ; relancer `./prepare-model.sh` |
| Chaque tour d'agent est lent (gros prefill à chaque fois) | correctif prefix cache absent | idem |
| `NV_ERR_NO_MEMORY` sous charge | marge mémoire épuisée | passer `--kv-cache-memory-bytes` à `4294967296` (4 GiB, 372K tokens) sur A et B |
| Out of memory au chargement ou au profiling | page cache plein ou autre process GPU | étape 8 : arrêter les autres modèles et vider le cache ; sinon baisser `--max-model-len` |
| Moteur tué pendant un très long prefill | pic mémoire de l'indexeur sparse (vllm#55569) | garder `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Appels d'outils jamais déclenchés | mauvais parser | `--tool-call-parser glm47` (pas `glm`) |
| Un nœud télécharge le modèle au démarrage | poids absents localement | étape 5 |
| Coupure ou ralentissement sous charge | thermique | étape 3, `systemctl status vllm-thermal` sur **les deux** machines |

---

## Fichiers du dépôt

| Fichier | Rôle |
|---|---|
| `docker-compose.yml` | service vLLM, identique sur les deux nœuds |
| `.env.example` | modèle de `.env` (rôle du nœud, IP, interfaces) |
| `prepare-model.sh` | poids, drafter, image et correctifs, à révisions figées |
| `netplan/40-cx7.yaml` | IP fixe du lien direct ConnectX-7 |
| `check-cx7.sh` | vérification du lien et des poids avant démarrage |
| `setup-thermal.sh`, `vllm-thermal.service` | limitation des fréquences GPU (GB10) |
