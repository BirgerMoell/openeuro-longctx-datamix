#!/usr/bin/env python3
"""Length-biased long-context sampler for the 64K / 128K stages.

Reads a Megatron .bin/.idx source, selects LONG documents in [min_len, max_len] (diverse,
seeded shuffle, taken to a token budget), and writes a new .bin/.idx via Megatron's
IndexedDatasetBuilder (correct format). This upsamples genuine long-range documents so the
long-context training stage sees full-window dependencies (fixes depth-0 retrieval weakness).

Run with --min-len 65536 for the 64K stage, --min-len 131072 for the 128K stage.
"""
import argparse, struct, sys, numpy as np, torch

def read_index(idx_path):
    with open(idx_path, "rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00", "bad idx magic"
        assert struct.unpack("<Q", f.read(8))[0] == 1, "bad idx version"
        code = struct.unpack("<B", f.read(1))[0]
        nseq = struct.unpack("<Q", f.read(8))[0]
        struct.unpack("<Q", f.read(8))  # document_count
        lengths = np.frombuffer(f.read(nseq * 4), dtype=np.int32).astype(np.int64)
        pointers = np.frombuffer(f.read(nseq * 8), dtype=np.int64)
    return code, lengths, pointers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-tokens", type=int, required=True)
    ap.add_argument("--min-len", type=int, default=131072)
    ap.add_argument("--max-len", type=int, default=1048576)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--megatron", default="/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM")
    a = ap.parse_args()
    sys.path.insert(0, a.megatron)
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder, DType

    code, lengths, pointers = read_index(a.src + ".idx")
    dtype = DType.dtype_from_code(code)
    itemsize = dtype().itemsize
    binmm = np.memmap(a.src + ".bin", dtype=dtype, mode="r")

    elig = np.where((lengths >= a.min_len) & (lengths <= a.max_len))[0]
    np.random.default_rng(a.seed).shuffle(elig)
    name = a.src.split("/")[-1]
    print(f"[{name}] eligible docs in [{a.min_len},{a.max_len}]: {len(elig)} "
          f"({lengths[elig].sum()/1e9:.2f}B tok available)", flush=True)

    builder = IndexedDatasetBuilder(a.out + ".bin", dtype=dtype)
    tok = 0; nd = 0
    for i in elig:
        if tok >= a.budget_tokens:
            break
        L = int(lengths[i]); off = int(pointers[i] // itemsize)
        builder.add_item(torch.from_numpy(np.asarray(binmm[off:off + L])))
        builder.end_document()
        tok += L; nd += 1
    builder.finalize(a.out + ".idx")
    print(f"[{name}] wrote {nd} docs / {tok/1e9:.2f}B tokens -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
