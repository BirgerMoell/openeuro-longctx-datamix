#!/usr/bin/env python3
"""Quantify long-range UNDERSAMPLING in the actual training mix.

For each source in the run's DATA mix, parse .idx doc lengths and compute the fraction of its
tokens that live in docs >= 64K and >= 128K. Weight by the mix weights to get the EFFECTIVE
fraction of training tokens (at the 64K and 128K stages) that carried genuine full-window
long-range structure. Multiply by the stage token budgets to get absolute "real long" tokens.
"""
import struct, numpy as np, re, sys

CAT = "/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue"
SCRIPT = "/scratch/project_465002530/users/bmoell/longctx-extend/extend_real_to128k.sh"
STAGE_BUDGET = {"64K": 2_000_000_000, "128K": 1_000_000_000}

def lengths(idx):
    with open(idx, "rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00"
        struct.unpack("<Q", f.read(8)); struct.unpack("<B", f.read(1))
        nseq = struct.unpack("<Q", f.read(8))[0]; struct.unpack("<Q", f.read(8))
        return np.frombuffer(f.read(nseq*4), dtype=np.int32).astype(np.int64)

# parse DATA="w path w path ..." from the run script
txt = open(SCRIPT).read()
m = re.search(r'DATA="([^"]+)"', txt, re.S)
toks = m.group(1).split()
pairs = [(float(toks[i]), toks[i+1]) for i in range(0, len(toks), 2)]
W = sum(w for w, _ in pairs)

eff64 = 0.0; eff128 = 0.0; rows = []
for w, path in pairs:
    idx = path + ".idx"
    try:
        L = lengths(idx)
    except Exception as e:
        rows.append((path.split("/")[-1], w/W, None, None)); continue
    tot = L.sum()
    f64 = L[L >= 65536].sum() / tot
    f128 = L[L >= 131072].sum() / tot
    nw = w / W
    eff64 += nw * f64; eff128 += nw * f128
    rows.append((path.split("/")[-1][:22], nw, f64, f128))

print(f"{'source':24} {'mixwt%':>7} {'%tok>=64K':>10} {'%tok>=128K':>11}")
for name, nw, f64, f128 in sorted(rows, key=lambda r: -r[1])[:12]:
    if f64 is None: print(f"{name:24} {100*nw:6.1f}%   (missing)"); continue
    print(f"{name:24} {100*nw:6.1f}% {100*f64:9.1f}% {100*f128:10.1f}%")
print("  ... (37 finepdfs langs + edu + code + math + web)")
print()
print(f"EFFECTIVE long-range token fraction in the mix:")
print(f"  64K stage:  {100*eff64:.1f}% of tokens are in docs >= 64K   -> the rest is packed-shorts")
print(f"  128K stage: {100*eff128:.1f}% of tokens are in docs >= 128K")
print()
print(f"ABSOLUTE genuine long-range tokens trained (budget x effective fraction):")
print(f"  64K stage:  {STAGE_BUDGET['64K']/1e9:.1f}B budget x {100*eff64:.0f}% = {STAGE_BUDGET['64K']*eff64/1e9:.2f}B real-long tokens")
print(f"  128K stage: {STAGE_BUDGET['128K']/1e9:.1f}B budget x {100*eff128:.0f}% = {STAGE_BUDGET['128K']*eff128/1e9:.2f}B real-long tokens")
