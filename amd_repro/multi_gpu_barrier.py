"""
    Copyright (c) 2024, Fireworks AI.
"""

import os
import torch
import torch.distributed as dist
import random

import amd_repro.lib

# Init torch.distributed.
dist.init_process_group(backend="nccl")
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(local_rank)

def run():
    # Create barrier instance.
    barrier = amd_repro.lib.create_barrier(
        device.index,
        local_rank,
        local_world_size,
    )
    torch.cuda.synchronize()

    # Exchange peer buffer handles.
    with torch.inference_mode(False), torch.no_grad():
        gloo_pg = dist.new_group(backend="gloo")
        handle = barrier.bufferHandle()
        handles = [torch.zeros_like(handle) for _ in range(local_world_size)]
        dist.all_gather(handles, handle, group=gloo_pg)
        barrier.setPeerBufferHandles(handles)

    for _ in range(1000):
        # Read and write some memory.
        # ATTENTION, it does seem to work if the following two lineas of code are removed.
        t = torch.randn((random.randint(0, 10000000),), device=device)
        t = t + t

        barrier.run()

i = 0
while True:
    if local_rank == 0:
        print("Running iter #", i)
    run()
    i += 1
