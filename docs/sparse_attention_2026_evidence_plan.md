# Evidence-based sparse-attention plan (August 2026)

## Decision

Keep the OpenEuroLLM Qwen3-style 9B GQA backbone and make attention switchable.  Do not replace
the deliverable architecture with KDA, NoPE, MLA, or a new recurrent model.  The immediate path is
a continued-pretraining conversion: preserve the dense checkpoint and add a detachable block/token
router that can be enabled only for super-long training and inference.

The passing 512K/CP16 round trip proved mechanics, but its 512-token support (current 256-token
block plus one remote block) was intentionally too small to justify a quality claim.  Recent
independent systems converge on an active budget near **2,048 tokens per query**.  The next run
therefore uses 128-token blocks and 16 selected blocks rather than scaling up the old 512-token
scaffold.

## What the latest primary sources actually show

| Work | Relevant result | What transfers to OpenEuroLLM | What does not transfer directly |
|---|---|---|---|
| [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392) and [official MSA kernels](https://github.com/MiniMax-AI/MSA) | Converts a pretrained GQA model using a detached KL indexer, dense indexer warm-up, 128-token blocks, 16 blocks (2,048 tokens), and a forced local block; reports near-GQA quality and a 1M production model | Closest architectural match: GQA, block selection, 2K budget, two-stage training | Released kernels require NVIDIA SM100/CUDA; the paper uses one selector per GQA group, while our current lightning indexer shares one selection |
| [LongCat Sparse Attention](https://arxiv.org/abs/2608.01662) and [official code/model](https://github.com/meituan-longcat/LongCat-2.0) | Uses 16 sink + 1,024 local + roughly 1,008 dynamic tokens; 2,048 total; trains at 128K then 512K; keeps the sparse/dense loss gap below 0.01 and evaluates HELMET | Confirms the 2K budget, progressive length curriculum, and need for local/sink coverage and retrieval evaluation | Its cross-layer and hierarchical indexing are different from our current block router; its hierarchical stage is inference-only |
| [HiLS-Attention](https://arxiv.org/abs/2607.02980) and [official implementation](https://github.com/Tencent-Hunyuan/HiLS-Attention) | End-to-end chunk retrieval under LM loss; 64-token chunks × top-32 = 2,048 tokens plus a 512-token local window; converts OLMo3-7B and evaluates through 256K/1M | Supports chunk routing, a local path, sampled synthetic retrieval during CPT, and learning selection before ultra-long scaling | Adds landmark tokens, query calibration, and HoPE; conversion uses 5B–50B tokens, much larger than our first calibration |
| [Native Sparse Attention](https://arxiv.org/abs/2502.11089) and [open Triton implementation](https://github.com/fla-org/native-sparse-attention) | Combines compressed global context, selected blocks, and a sliding window with native sparse training | Reinforces multi-path global/local coverage and hardware-aligned blocks | This is a new attention module, not a low-risk late drop-in for our current checkpoint |
| [DeepSeek-V3.2 DSA](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | Dense indexer warm-up followed by sparse joint training with detached teacher/KL | This is the foundation already implemented in `scripts/dsa` | Official kernels/layout target MLA and NVIDIA, not our GQA/MI250X training stack |
| [Inkling](https://huggingface.co/thinkingmachines/Inkling), [config](https://huggingface.co/thinkingmachines/Inkling/blob/main/config.json), and [Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/inkling/modeling_inkling.py) | 1,048,576-token model with 55 local layers, 11 periodic global layers, and a 512-token local window | Strong evidence that cheap locality in most layers plus recurring global access is a viable 1M pattern | It is fixed hybrid local/global attention with learned relative bias, not DSA, and is not a dense-checkpoint conversion recipe |
| [Kimi K3](https://arxiv.org/abs/2607.24653) | Progresses from 8K to 64K, then 256K to 1M during cooldown; emphasizes coherent long documents and synthesized tasks whose evidence is distributed across the whole sequence | Data curriculum and staged length growth apply directly | K3 uses a 3:1 hybrid of recurrent KDA and global Gated MLA with NoPE; it is not a sparse-softmax drop-in |

The consistent lesson is that sparse training is not just “top-k instead of dense.”  Successful
systems combine a trained selector, a roughly 2K support, reliable local/prefix access, a warm-up or
conversion phase, coherent long data, and explicit long-context evaluation.  Training loss alone is
not a sufficient gate.

## Completed 512K calibration

**First launch outcome:** LUMI job `21050508`, submitted on 2026-08-12 from immutable source commit
`84feff7`, failed after two seconds in the launcher's required-path check. The shared checkout
`/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM` had been removed before
the allocation began. Preflight, data loading, model loading, and training never started; no output
artifact was created. The pinned revision `b359462c12858cedd2238a22eca0dca7aa6b8872` remains
available from the official NVIDIA Megatron-LM repository. Before a replacement launch, create an
immutable Birger-owned checkout of that exact revision, verify it with the existing preflight, and
pass it explicitly through `MEGATRON_ROOT`. Do not silently use a different Megatron revision.

**Repair and replacement:** the exact upstream revision was staged at
`/scratch/project_465002530/users/bmoell/deps/NVIDIA-Megatron-LM-b359462c`. One-node preflight job
`21265221` completed successfully in 63 seconds and verified the revision, overlay/module imports,
warm checkpoint, 48 data pairs, CP/block geometry, and dataset surplus. The unchanged experiment
was resubmitted as job `21265492` with `MEGATRON_ROOT` pointing to this owned checkout. That job
completed updates 301–336 and a complete iteration-336 checkpoint. Its fresh phase-2 process then
stalled on the first update in one CP RCCL all-gather; seven dependent TP lanes timed out. The same
collective had run throughout phase 1, and there was no model, data, numerical, or checkpoint
error, so this was treated as a transient communicator failure. Recovery job `21284719` loaded the
verified iteration-336 state and completed the experiment without repeating phase 1.

The first evidence-calibrated run is deliberately bounded but is real full-parameter continued
pretraining, not another two-update plumbing test:

- source: the 256K/θ=64M OpenEuroLLM checkpoint plus the completed all-layer frozen indexer
  checkpoint at iteration 300;
- data: the checksum-verified 48-prefix superlong-v2 Megatron blend;
- global context 524,288, CP16, local context 32,768, TP8, 128 ranks / 16 nodes;
- θ=128M;
- all 36 layers sparse;
- router: 128-token blocks, current block forced, 15 learned earlier blocks, 2,048 selected tokens;
- detached selected-set KL remains active while the LM objective updates the base model;
- 72 adaptation updates in two fresh 36-update processes, then a third fresh-process reload and
  finite update (73 updates / 38.27M training tokens total);
- each 36-update phase samples an exact dense attention teacher for 64 queries in every layer and
  reports attention-mass recall at k=512/1,024/2,048, including global-position quartiles;
- checkpoint after each 36-update phase and after the final reload update.

The observed steady update time was about 81.7 seconds (roughly 50.1 tokens/s/GPU). Peak memory was
about 17.7 GiB allocated and 25.7 GiB reserved per rank. Scheduler allocations across the failed
two-second launch, one-node preflight, phase-1/transient-failure job, and recovery job consumed
about **315.0 GPU-hours**. This includes the roughly 33-minute failed communicator wait; the
successful recovery allocation itself was 128 GPU-hours.

This router is best described as **MSA/LSA-informed DSA**, not a reproduction of either paper.  It
still shares one block selection across attention heads and uses mean block summaries.  Those are
the next algorithmic changes if recall is weak.

### Terminal result: mechanics pass, router quality fails

All 73 intended updates completed with finite losses and gradients and zero skipped or NaN
iterations. Iterations 336, 372, and 373 each contain `.metadata`, `common.pt`, and 256 distributed
shards. The third fresh process loaded iteration 372, including the distributed optimizer, then
completed update 373 and emitted `DSA 512K k2048 calibration PASS`.

| Signal | Phase 1: 301–336 | Phase 2: 337–372 | Reload update 373 |
|---|---:|---:|---:|
| Mean LM loss | 2.2370 | 2.0115 | 2.1218 |
| Final LM loss | 2.5338 | 2.0425 | 2.1218 |
| Mean indexer loss | 0.7122 | 0.6753 | 0.6328 |
| Final indexer loss | 0.6750 | 0.6354 | 0.6328 |
| Main-model probe norm | 2.2478 | 3.4279 | 1.8804 |
| Indexer probe norm | 0.00700 | 0.01081 | 0.00703 |
| Skipped / NaN updates | 0 / 0 | 0 / 0 | 0 / 0 |

The exact sampled dense-teacher recall cycles do **not** pass the quality criterion:

| Retained tokens | Phase-1 mean / position quartiles | Phase-2 mean / position quartiles |
|---|---|---|
| 512 | 0.057 / [0.138, 0.042, 0.029, 0.021] | 0.055 / [0.134, 0.039, 0.027, 0.019] |
| 1,024 | 0.072 / [0.173, 0.053, 0.036, 0.027] | 0.067 / [0.163, 0.048, 0.034, 0.025] |
| 2,048 | 0.103 / [0.220, 0.082, 0.061, 0.047] | 0.092 / [0.206, 0.069, 0.053, 0.041] |

Mean top-2,048 recall fell by 0.0104 (about 10% relative), with 25 of 36 layers declining. The
router is strongly position-biased: even at k=2,048, the last three quartiles capture only about
4–7% attention mass after phase 2. This bounded run therefore validates sustained 512K training
and recovery, but its artifact is **not accepted as a quality-successful sparse conversion**.

## Pass, stop, and expansion criteria

The calibration artifact was evaluated against these gates:

1. **Passed after one documented transient recovery:** all 73 updates have finite LM and indexer
   losses, finite nonzero main/indexer probes, and no skipped or NaN iterations. One CP communicator
   stalled between phases and was safely resumed from checkpoint 336;
2. **Passed:** iterations 336, 372, and 373 are complete distributed checkpoints, with iteration
   373 produced only after loading full optimizer/RNG/scheduler state in a fresh process;
3. **Failed:** top-2,048 attention-mass recall was reported for every layer in both phases, but was
   low, position-biased, and declined during adaptation;
4. **Passed:** LM loss remained finite without a sustained upward break; and
5. **Not run because gate 3 failed:** held-out 512K loss and retrieval testing remain mandatory for
   any repaired candidate before it is called a quality success.

Do not keep training through deterministic numerical failure or a clearly collapsed router.  A
mechanically successful but low-recall result means the next experiment is a router fix (per-GQA
selection, an explicit 512–1,024 local window/sink, or longer 128K/256K indexer adaptation), not a
larger token budget.

Do **not** launch the proposed 0.1B-token run from iteration 373. The next experiment must repair
the router: add a guaranteed 512–1,024-token local path and sink/prefix coverage, make selection
per GQA group instead of shared across all heads, and give the indexer a longer 128K/256K
dense-teacher adaptation before another bounded 512K test. Only a fixed-batch dense/sparse loss and
retrieval gate with materially better per-quartile recall should unlock a longer 512K schedule.

## Route to 1M–2M

The current CP bridge all-gathers global K/V and indexer keys on every CP rank.  It is acceptable at
512K but should not be doubled repeatedly.  Before 1M:

1. replace replicated global K/V with selected-block/row exchange or an owner-compute sparse
   attention collective;
2. make selection per GQA group, or prove the shared selector has comparable recall;
3. preserve an explicit local/prefix path if the learned selector does not recover it reliably;
4. fuse the Python head/query launch structure into a ROCm block-sparse kernel;
5. build sparse CP-aware teacher-forced retrieval and eventually generation evaluation; and
6. progress 512K → 1M → 2M on the existing coherent/synthetic superlong mixture, with a held-out
   short-context regression suite at every stage.

Kimi K3 and Inkling demonstrate that 1M is feasible, but through very different attention
architectures.  For this project, the shortest credible route is to make the GQA-compatible 2K
block router work at 512K first, then remove the replicated-state transport bottleneck.
