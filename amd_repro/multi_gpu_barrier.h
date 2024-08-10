/*
 * Copyright (c) 2024, Fireworks AI.
 */
#pragma once

#include <ATen/ATen.h>

#include <memory>
#include <vector>

namespace amd_repro {

constexpr int MAX_RANKS = 8;

struct Params {
  uint64_t* barrier_flag_ptr;
  uint64_t* peer_barrier_ptrs[MAX_RANKS];
  int rank;
  int num_ranks;
};

class MultiGpuBarrier {
 public:
  MultiGpuBarrier(int device, int rank, int num_ranks);
  ~MultiGpuBarrier();

  void run();

  at::Tensor bufferHandle();

  void setPeerBufferHandles(const std::vector<at::Tensor>& handles);

 private:
  c10::DeviceIndex device_;
  Params params_;
  at::Tensor flag_;
};

std::shared_ptr<MultiGpuBarrier> create_barrier(int64_t device, int64_t rank,
                                                int64_t num_ranks);

}  // namespace amd_repro
