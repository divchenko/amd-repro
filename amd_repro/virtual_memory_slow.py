"""
    Copyright (c) 2024, Fireworks AI.
"""

import torch
import time

import amd_repro.lib

VIRTUAL_SIZE =  1 << 38
device = torch.cuda.current_device()

start = time.time()

ptr = amd_repro.lib.reserve_virtual_memory(device, VIRTUAL_SIZE)

print("Total time", int(time.time() - start), "secs")
