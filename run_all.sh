#!/bin/sh
# Every command behind the numbers in the README and stage descriptions. Usage: ./run_all.sh [seed]
set -e
SEED=${1:-0}
P=solutions/python/17-mtp/code/your_program.sh
D=data/tinyshakespeare.txt
OUT=runs/seed$SEED
mkdir -p "$OUT"
if [ "$SEED" = 0 ]; then
  $P train-dense $D --steps 3000 --out-dir "$OUT"
fi
$P compare $D --seed "$SEED" --steps 3000 --variants dense,dense-wide,moe-none,moe-aux,moe-bias,moe-bias-fine --out-dir "$OUT"
$P route-lab $D --seed "$SEED" --ckpt "$OUT/moe-bias_s$SEED.pt" --out-dir "$OUT"
$P ablate $D --seed "$SEED" --ckpt "$OUT/moe-bias_s$SEED.pt" --out-dir "$OUT"
$P route-lab $D --seed "$SEED" --ckpt "$OUT/moe-none_s$SEED.pt" --out-dir "$OUT/none"
$P collapse $D --seed "$SEED" --steps 1500 --out-dir "$OUT"
$P mtp $D --seed "$SEED" --steps 3000 --out-dir "$OUT"
