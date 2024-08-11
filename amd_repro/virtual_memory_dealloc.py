"""
    Copyright (c) 2024, Fireworks AI.
"""

import torch

import amd_repro.lib

VIRTUAL_SIZE = 1 << 24
ALLOC_SIZE = 1 << 20
device = torch.cuda.current_device()

free_mem_1, _ = torch._C._cudart.cudaMemGetInfo(device)
ptr = amd_repro.lib.reserve_virtual_memory(device, VIRTUAL_SIZE)
free_mem_2, _ = torch._C._cudart.cudaMemGetInfo(device)
print("Allocated mem", free_mem_1 - free_mem_2)

free_mem_1, _ = torch._C._cudart.cudaMemGetInfo(device)
handle = amd_repro.lib.allocate(device, ptr, ALLOC_SIZE)
free_mem_2, _ = torch._C._cudart.cudaMemGetInfo(device)
print("Allocated mem", free_mem_1 - free_mem_2)

free_mem_1, _ = torch._C._cudart.cudaMemGetInfo(device)
amd_repro.lib.deallocate(ptr, handle, ALLOC_SIZE)
free_mem_2, _ = torch._C._cudart.cudaMemGetInfo(device)
print("Deallocated mem", free_mem_2 - free_mem_1)
