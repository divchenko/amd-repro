"""
    Copyright (c) 2024, Fireworks AI.
"""

import torch

import amd_repro.lib

VIRTUAL_SIZE = 1 << 24
ALLOC_SIZE = 1 << 20
device = torch.cuda.current_device()


ptr = amd_repro.lib.reserve_virtual_memory(device, VIRTUAL_SIZE)
handle = amd_repro.lib.allocate(device, ptr, ALLOC_SIZE)
amd_repro.lib.export_import(handle)
