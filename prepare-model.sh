#!/usr/bin/env bash
# Prépare tout ce que le compose monte depuis l'hôte, sur CE nœud :
#   - poids nvidia/GLM-5.3-Flash-NVFP4 (~205 Go) et drafter DFlash2 (2,2 Go)
#   - l'image patchée sm121
#   - les deux correctifs vLLM montés par-dessus l'image (recette tonyd2wild)
# Idempotent : relancer ne retélécharge que ce qui manque.
# Usage: ./prepare-model.sh
set -euo pipefail

MODELS=/home/mak/ai/models
GLM=/home/mak/ai/glm53
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2

# Révisions figées : les deux nœuds doivent servir exactement les mêmes fichiers
WEIGHTS_REV=da920bb0b9f4a06727223a349e55468e38352348   # nvidia/GLM-5.3-Flash-NVFP4
DRAFTER_REV=bf582e4eacc1810f76656d1811693ff6c6737d2a   # incoai/GLM-5.3-Flash-DFlash2
RECIPE_REV=d061f26ad3ec5c3c04f64aad7516dbe62a8aa6af    # tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark
RECIPE_RAW=https://raw.githubusercontent.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/$RECIPE_REV

mkdir -p "$GLM/patches" "$GLM/cache/flashinfer" "$GLM/cache/tilelang" "$GLM/cache/triton"

echo "=== Poids et drafter"
hf download nvidia/GLM-5.3-Flash-NVFP4 --revision "$WEIGHTS_REV" --local-dir "$MODELS/GLM-5.3-Flash-NVFP4"
# Licence CC-BY-NC-ND 4.0 : usage non commercial uniquement
hf download incoai/GLM-5.3-Flash-DFlash2 --revision "$DRAFTER_REV" --local-dir "$MODELS/GLM-5.3-Flash-DFlash2"

echo "=== Image"
docker pull "$IMAGE"

echo "=== Correctif top-k SM121 (sans lui, le moteur meurt au-delà de ~24K de contexte)"
curl -fsSL "$RECIPE_RAW/docker/sparse_attn_indexer_kpool_sm121.py" -o "$GLM/patches/sparse_attn_indexer_kpool.py"

echo "=== Correctif prefix cache du drafter (#18), appliqué dans un conteneur jetable"
curl -fsSL "$RECIPE_RAW/docker/dflash2-overlay/patch_prefix_cache_draft_group.py" -o "$GLM/patches/patch_prefix_cache_draft_group.py"
docker run --rm --entrypoint bash -v "$GLM/patches:/patches" "$IMAGE" -c '
  set -e
  python3 /patches/patch_prefix_cache_draft_group.py
  cp /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py /patches/kv_cache_coordinator.py'

echo "[OK] Tout est prêt dans $MODELS et $GLM"
