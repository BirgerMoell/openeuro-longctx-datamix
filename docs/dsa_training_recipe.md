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
