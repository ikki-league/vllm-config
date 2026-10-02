#!/usr/bin/env bash
# Prépare les poids GLM-5.3 Flash NVFP4 (NVIDIA) dans /home/mak/ai/models.
# À lancer sur CHAQUE nœud (ou une fois sur le head puis rsync, voir README).
# Usage: ./prepare-model.sh [--mtpfix]
#
# --mtpfix : exclut aussi le parallel_lm_head de la couche MTP de la
# quantification NVFP4. Contournement de vllm#57532, inutile avec l'image du
# compose (correctif vllm#55442 inclus) ; à n'appliquer que si le démarrage
# échoue sur "NVFP4 weight_scale for layer 'parallel_lm_head' was never loaded".
set -euo pipefail

REPO=nvidia/GLM-5.3-Flash-NVFP4
REVISION=da920bb0b9f4a06727223a349e55468e38352348
DEST=/home/mak/ai/models/GLM-5.3-Flash-NVFP4

if [ "${1:-}" != "--mtpfix" ]; then
  # ~205 Go ; reprend là où il s'est arrêté si interrompu
  hf download "$REPO" --revision "$REVISION" --local-dir "$DEST"
  echo "[OK] $REPO@${REVISION:0:7} dans $DEST ($(du -sh "$DEST" | cut -f1))"
  exit 0
fi

python3 - "$DEST" <<'EOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
extra = ["parallel_lm_head", "*parallel_lm_head*"]
for name, path in [("hf_quant_config.json", ("quantization", "exclude_modules")),
                   ("config.json", ("quantization_config", "ignore"))]:
    f = d / name
    cfg = json.loads(f.read_text())
    lst = cfg[path[0]][path[1]]
    added = [e for e in extra if e not in lst]
    lst.extend(added)
    f.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[OK] {name}: {'ajouté ' + ', '.join(added) if added else 'déjà patché'}")
EOF
