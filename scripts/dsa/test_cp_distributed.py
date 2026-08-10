#!/usr/bin/env python3
"""Two-rank autograd gate for the DSA context-parallel global gather."""

import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cp_utils import cp_global_positions, gather_global_sequence


def _run(rank, world, backend, init_method=None):
    if init_method is None:
        dist.init_process_group(backend)
    else:
        dist.init_process_group(
            backend, init_method=init_method, rank=rank, world_size=world
        )
    try:
        world = dist.get_world_size()
        rank = dist.get_rank()
        if world < 2:
            raise RuntimeError("distributed CP test requires at least two ranks")
        if backend == "nccl":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        local_length = 8
        positions = cp_global_positions(local_length, world, rank, device=device)
        local = positions.to(torch.float32).view(local_length, 1).requires_grad_()
        global_tensor = gather_global_sequence(local, dist.group.WORLD)
        expected = torch.arange(
            local_length * world, device=device, dtype=torch.float32
        ).view(-1, 1)
        torch.testing.assert_close(global_tensor, expected, rtol=0, atol=0)

        rank_weight = float(rank + 1)
        (global_tensor * rank_weight).sum().backward()
        expected_gradient = world * (world + 1) / 2
        torch.testing.assert_close(
            local.grad,
            torch.full_like(local, expected_gradient),
            rtol=0,
            atol=0,
        )
        dist.barrier()
        if rank == 0:
            print(
                "PASS distributed CP gather: global token order and summed autograd "
                f"across {world} ranks"
            )
    finally:
        dist.destroy_process_group()


def _free_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--spawn-procs",
        type=int,
        default=0,
        help="Spawn this many local processes instead of relying on torchrun/Slurm.",
    )
    args = parser.parse_args()
    if args.spawn_procs:
        init_method = f"tcp://127.0.0.1:{_free_local_port()}"
        mp.spawn(
            _run,
            args=(args.spawn_procs, args.backend, init_method),
            nprocs=args.spawn_procs,
            join=True,
        )
    else:
        _run(
            int(os.environ.get("RANK", "0")),
            int(os.environ.get("WORLD_SIZE", "1")),
            args.backend,
        )


if __name__ == "__main__":
    main()
