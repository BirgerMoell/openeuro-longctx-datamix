# Current RoPE θ values (reference)

The RoPE base-frequency (θ / `rotary_base`) used across our base + long-context extension stages, and
the published models. Method background: `docs/handoff_longcontext_for_jouni.md`.

## Model geometry (fixed)
- `head_dim = 128`, **full rotary** (`partial_rotary_factor = 1.0`, rotary_dim = 128)
- 36 layers, 32 Q heads / 8 KV heads (GQA), hidden 4096
- (Contrast Qwen3.5: head_dim 256, partial rotary 0.25 → rotary_dim 64 — see `rope_theta_vs_qwen35.md`)

## θ by training stage (staged native ABF)
| stage | seq length | **θ (rotary_base)** |
|---|---|---|
| base pretraining | 4K | **100,000 (100k)** |
| extend | 16K | 500,000 |
| extend | 32K | 1,000,000 |
| extend | 64K | 2,000,000 |
| extend | **128K** | **32,000,000 (32M)** |
| extend | **256K** | **64,000,000 (64M)** |

θ jumps sharply at 128K — that is the **θ-law** (critical θ ≈ doubles per context-length octave),
not a smooth ramp.

## θ by published model
| model | context | θ | base |
|---|---|---|---|
| `oellm-9b-128k-theta32m-v3` | 128K | 32M | 0.6T |
| `oellm-9b-128k-theta32m-prelude` | 128K | 32M | 1T |
| `oellm-9b-256k-theta64m-prelude` | 256K | 64M | 1T |

## The evidence behind these values (128K θ-sweep, our depth-0 metric)
| θ @128K | depth-0 (base-LM forced-choice NIAH, in-context distractors) |
|---|---|
| 2M / 5M / 8M | ≈ 0% |
| 16M | 90% |
| **32M** | **100%** ← chosen |

So 32M is the **minimum that saturated our depth-0 bar at 128K**; 64M for 256K is the doubling-law
extrapolation (confirmed *sufficient*: depth-0 ≈ 93–100%, never shown *minimal*). Full analysis of
whether these are over-scaled (vs Qwen's 10M): `docs/rope_theta_vs_qwen35.md`. Suggestions for the
next run: `docs/rope_theta_new_run_suggestions.md`.
