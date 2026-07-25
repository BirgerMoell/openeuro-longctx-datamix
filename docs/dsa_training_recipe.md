# DSA training recipe (stable) — from DeepSeek-V3.2 + our findings

**Our instability bug:** we ran SPARSE (top-k=2048) from step 1 with a RANDOM indexer → the model
attends to garbage tokens → noisy loss (1.4–2.5), grad norm ~5000. Fix = dense warm-up.

## DeepSeek-V3.2 DSA recipe (confirmed)
1. **Dense warm-up:** attention kept DENSE (full), **freeze all model params except the indexer**,
   train indexer with **KL loss**. 1000 steps × (16 seqs × 128K) = **2.1B tokens**. Context 128K.
2. **Sparse stage:** introduce top-k selection (k=2048), train **model + indexer**. 15000 steps ×
   (480 × 128K) = **943.7B tokens**. (GLM-5 got away with ~20B for the sparse stage — far less.)

## Our adapted recipe (continue from ckpt_262144)
- **Dense warm-up:** set `DSA_TOPK = seq_length` (sparse-attn == dense → model UNPERTURBED →
  lm loss stays ~1.3, STABLE) + `DSA_LOSS_COEFF>0` (KL trains indexer). Model not frozen (our
  autoscaler needs the main-loss backward; dense attn means the model barely moves anyway). Our
  reference attn is O(L²)-memory → warm at 16–32K (indexer length-generalizes); ~0.5–2B tokens.
- **Sparse adapt:** `DSA_TOPK=2048`, `DSA_SPARSE_RUN=1` (Triton attn + chunked indexer + KL off),
  train the model to adapt. Watch lm loss recover toward the dense value.
- **Stability levers if still noisy:** larger effective batch (grad-accum, not tiny gbs=8);
  gentle LR; ramp k down gradually (full → 4096 → 2048) rather than a hard switch.

## Why warm short, run long
The indexer learns "which tokens matter" — length-generalizes (like the θ-law). So warm at 16–32K
(cheap, O(L²) fits), then INFERENCE/sparse-run at 512K–2M with the Triton O(L·k) kernel.

## Update (2026-07-25): indexer LR is the key lever (from live recall + literature)
Our first warm-up (LR 1e-4) had recall creep 0.37→0.40 over 46 iters — learning, but ~50× too slow.
**DeepSeek/GLM train the indexer at ~5e-3** (vs model ~1e-5). During DENSE warm-up the model is
unperturbed (attention is full), so a high global LR mostly drives the indexer. Fix: warm up at
**LR 5e-3**. Literature cross-checks (all confirm dense-warmup-then-sparse):
- **MoBA**: "indexer warmup runs full attention before switching to sparse → gives a stable target
  to imitate; same recipe converts a pretrained dense checkpoint into a sparse one." (= our exact plan)
- **InfLLM-V2** (dense-sparse switchable): sparse adaptation needs LITTLE data — cheap.
- **Kimi Linear**: hybrid ratio matters (they use 3 linear : 1 full) — supports per-layer hybrid.
- Refs: arXiv:2510.26692 (Kimi Linear), arXiv:2509.24663 (InfLLM-V2), MoonshotAI/MoBA.
