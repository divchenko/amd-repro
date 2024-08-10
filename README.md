# Repros of Issues on AMD GPUS

## Building
It requires ROCm image with PyTorch (new versions are desired, i.e. >= 2.5).
```
python setup.py develop
```
See individual issue for running instructions.

## Multi-GPU barrier.
It relies on GPU peer access over unified memory.
Each rank allocates an array of flags, which is used to notify other ranks.

Run using:
```
torchrun --nproc-per-node 8 amd_repro/multi_gpu_barrier.py
```
It hangs after a few iterations despite all attempts to invalidate and write back caches.