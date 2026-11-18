# The Start of the Window — technical report

*The Start of the Window: RoPE's Critical Dimension Predicts the Base Frequency Needed for 128K–256K Context Extension.*

- `rope-base-depth0.tex` — the report; `rope-base-depth0.pdf` — compiled PDF.
- `analysis.py` — regenerates every number, table and figure from the raw trials.
- `raw/lumi/eval_niah_128k_<job>/` — per-trial NIAH results (with candidate log-likelihoods) copied from LUMI, one directory per Slurm evaluation job; `raw/lumi/*.sbatch|*.sh` — the exact launch scripts. The mapping from job to checkpoint is the `RUNS` table in `analysis.py`.
- `generated/` (LaTeX macros and tables) and `figures/` are outputs of `analysis.py`.

```bash
uv run --no-project --python 3.12 --with numpy --with matplotlib python paper/analysis.py
cd paper && tectonic -X compile rope-base-depth0.tex
```
