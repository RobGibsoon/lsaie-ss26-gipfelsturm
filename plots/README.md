# Plots

All plots are generated from parsed run JSONs. Raw logs live in `logs/logs/`, parsed JSONs in the comparison directories. See `plot/` for the scripts.

---

## fp8-760m-recipe-comparison.png

Violin plot comparing all 6 FP8 recipes against a BF16 baseline for the 760m model at 1 node (50 steps). Ordered from slowest (bottom) to fastest (top) by median tokens/sec/GPU.

**Data:** `fp8-760m-recipe-comparison/` — 7 JSONs (1 BF16 baseline + 6 FP8 variants: hybrid/e4m3 × delayed/tensorwise/blockwise)

**Command:**
```bash
python3 plot/violinplot.py \
  -f fp8-760m-recipe-comparison/ \
  --metrics tokens_per_sec_per_gpu \
  -t "760M FP8 recipe comparison (1 node, 50 steps)" \
  -o plots/fp8-760m-recipe-comparison.png
```

---

## fp8-8b-comparison.png

Violin plot comparing FP8 vs BF16 configurations for the 8b model at 1 node (50 steps), varying TP (1/2) and MBS (1/2). Ordered from slowest (bottom) to fastest (top). Includes an OOM marker for fp8-delayed-tp1-mbs2, which is why the best FP8 TP=1 configs use MBS=1.

**Data:** `fp8-8b-comparison/` — BF16 (TP1/TP2, MBS2), FP8 delayed/tensorwise (TP1 MBS1, TP2 MBS2), OOM stub for fp8-delayed-tp1-mbs2

**Command:**
```bash
python3 plot/violinplot.py \
  -f fp8-8b-comparison/ \
  --metrics tokens_per_sec_per_gpu \
  -t "8B FP8 vs BF16 comparison (1 node, 50 steps)" \
  -o plots/fp8-8b-comparison.png
```

---

## bf16-strong-scaling.png

Strong scaling plot for BF16 across three model sizes (125m, 760m, 8b) from 1 to 8 nodes (4–32 GPUs). One line per model size. Shows how per-GPU throughput degrades as GPUs increase with a fixed global batch size (GBS=256).

**Data:** `bf16-scaling/125m-{1,2,4,8}n.json`, `760m-{1,2,4,8}n.json`, `8b-{1,2,4,8}n.json`

**Command:**
```bash
python3 plot/scatter_plot.py --mode strong \
  -f bf16-scaling/125m-1n.json bf16-scaling/125m-2n.json bf16-scaling/125m-4n.json bf16-scaling/125m-8n.json \
     bf16-scaling/760m-1n.json bf16-scaling/760m-2n.json bf16-scaling/760m-4n.json bf16-scaling/760m-8n.json \
     bf16-scaling/8b-1n.json   bf16-scaling/8b-2n.json   bf16-scaling/8b-4n.json   bf16-scaling/8b-8n.json \
  --metric tokens_per_sec_per_gpu --ylabel "Tokens / sec / GPU" \
  --ideal \
  -t "Strong scaling: BF16 (125m, 760m, 8b)" \
  -o plots/bf16-strong-scaling.png
```

---

## 8b-bf16-vs-fp8-scaling.png

Strong scaling plot for the 8b model comparing BF16 (MBS=1) against FP8 hybrid-delayed (MBS=1) from 1 to 8 nodes. Shows that FP8 maintains a consistent throughput advantage over BF16 across all node counts, though both degrade similarly with scale.

**Data:** `bf16-scaling/8b-{1,2,4,8}n.json` (BF16), `bf16-scaling/8b-fp8-{1,2,4,8}n.json` (FP8 hybrid-delayed)

**Command:**
```bash
python3 plot/scatter_plot.py --mode strong \
  -f bf16-scaling/8b-1n.json bf16-scaling/8b-2n.json bf16-scaling/8b-4n.json bf16-scaling/8b-8n.json \
     bf16-scaling/8b-fp8-1n.json bf16-scaling/8b-fp8-2n.json bf16-scaling/8b-fp8-4n.json bf16-scaling/8b-fp8-8n.json \
  --metric tokens_per_sec_per_gpu --ylabel "Tokens / sec / GPU" \
  --ideal \
  -t "8B strong scaling: BF16 vs FP8" \
  -o plots/8b-bf16-vs-fp8-scaling.png
```

---

## bf16-scaling.png

*(Superseded by `bf16-strong-scaling.png` — same data, earlier iteration.)*

## fp8-vs-bf16-size-sweep
Good coverage. BF16 has 5 sizes (125m, 760m, 1.5b, 3b, 8b), FP8 has 4 (760m, 1.5b, 3b, 8b). 

For some reason, when rerunning FP8 125m I had vastly different performance.

python3 plot/scatter_plot.py --mode size-sweep \
  -f fp8-vs-bf16-size-sweep/ \
  --metric tokens_per_sec_per_gpu \
  --ylabel "Tokens / sec / GPU" \
  -t "FP8 vs BF16 throughput across model sizes (1 node, TP=1)" \
  -o plots/fp8-vs-bf16-size-sweep.png

## fp8-vs-bf16-val-loss

Shows train performance of fp8 vs bf16 on 8b model.

python3 plot/loss.py fp8-vs-bf16-val-loss/ -t "Progression of validation loss (8b parameters)" -o plots/fp8-vs-bf16-val-loss.png