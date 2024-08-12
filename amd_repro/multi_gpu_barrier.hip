/*
 * Copyright (c) 2024, Fireworks AI.
 */

#include <ATen/ATen.h>
#include <ATen/hip/HIPContextLight.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

#include "multi_gpu_barrier.h"

namespace amd_repro {

namespace {
constexpr int NUM_BLOCKS = 304;
constexpr int BLOCK_SIZE = 64;

inline __device__ void st_flag_release(uint64_t flag, uint64_t* flag_addr) {
  __hip_atomic_exchange(flag_addr, flag, __ATOMIC_RELEASE,
                        __HIP_MEMORY_SCOPE_SYSTEM);

  // ATTENTION it works just fine on NVIDIA using 
  /*
    asm volatile("st.global.release.sys.b64 [%1], %0;" ::"l"(flag), "l"(flag_addr));
  */
                        
  // Another option below.
  // __hip_atomic_* do not produce 'nt' flag, which according to docs is needed
  // for LLC flush.
    /*
    __asm__ volatile(
      "global_store_dwordx2 %1, %0, off sc0 sc1 nt \n"
      "buffer_wbl2 sc0 sc1 \n" ::"v"(flag),
      "v"(flag_addr));
    */
}

inline __device__ void ld_flag_acquire(uint64_t& flag, uint64_t* flag_addr) {
  flag =
      __hip_atomic_load(flag_addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);

  // ATTENTION it works just fine on NVIDIA using 
  /*
    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(flag) : "l"(flag_addr));
  */
 
  // Another option below.
  // __hip_atomic_* do not produce 'nt' flag, which according to docs is needed
  // for LLC flush.
    /*
    __asm__ volatile(
        "buffer_inv sc0 sc1 \n"
        "global_load_dwordx2 %0, %1, off sc0 sc1 nt \n"
        "s_waitcnt vmcnt(0) \n"
        : "=v"(flag)
        : "v"(flag_addr));
    */
}

__global__ void multi_gpu_barrier_kernel(Params params) {
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  // Read flag.
  uint64_t barrier_flag = *(params.barrier_flag_ptr + bidx);

  // Barrier.
  if (tidx < params.num_ranks) {
    // The 1st block notifies the other ranks.
    st_flag_release(barrier_flag,
                    params.peer_barrier_ptrs[tidx] + bidx * MAX_RANKS + params.rank);

    // Busy-wait until all ranks are ready.
    uint64_t rank_barrier;
    uint64_t* peer_barrier_d = params.peer_barrier_ptrs[params.rank] + bidx * MAX_RANKS + tidx;
    int i = 0;
    do {
        ld_flag_acquire(rank_barrier, peer_barrier_d);
        i += 1;
        if (i == 10000 && params.rank == 1) {
          printf("! %d %d %d %d %d \n", params.rank, bidx, tidx, barrier_flag, rank_barrier);
        }
    } while (rank_barrier < barrier_flag);
  }

  __syncthreads();

  // Increment flag.
  if (tidx == 0) {
    *(params.barrier_flag_ptr + bidx) = barrier_flag + 1;
  }
}

bool isPeerAccessAvailable(std::vector<int64_t> peer_devices) {
  int peer_access_available = 0;
  for (const auto di : peer_devices) {
    const c10::DeviceGuard device_guard(
        {c10::kCUDA, static_cast<c10::DeviceIndex>(di)});
    for (const auto dj : peer_devices) {
      if (di == dj) {
        continue;
      }
      C10_HIP_CHECK(hipDeviceCanAccessPeer(&peer_access_available, di, dj));
      if (!peer_access_available) {
        return false;
      }
    }
  }
  return true;
}

void enablePeerAccess(std::vector<int64_t> peer_devices) {
  for (const auto di : peer_devices) {
    const c10::DeviceGuard device_guard(
        {c10::kCUDA, static_cast<c10::DeviceIndex>(di)});
    for (const auto dj : peer_devices) {
      if (di == dj) {
        continue;
      }
      C10_HIP_CHECK(hipDeviceEnablePeerAccess(dj, 0));
      const auto err = hipGetLastError();
      if (err != hipErrorPeerAccessAlreadyEnabled) {
        C10_HIP_CHECK(err);
      }
    }
  }
}

}  // namespace

MultiGpuBarrier::MultiGpuBarrier(int device, int rank, int num_ranks)
    : device_(device) {
  params_.rank = rank;
  params_.num_ranks = num_ranks;
  const c10::DeviceGuard device_guard(
      {c10::kCUDA, static_cast<c10::DeviceIndex>(device)});

  // Allocate barrier buffer.
  constexpr auto barrier_size = NUM_BLOCKS * MAX_RANKS * sizeof(uint64_t);
  auto peer_barrier_ptr =
      reinterpret_cast<void**>(&params_.peer_barrier_ptrs[rank]);
  C10_HIP_CHECK(hipMalloc(peer_barrier_ptr, barrier_size));
  C10_HIP_CHECK(hipMemset(*peer_barrier_ptr, 0, barrier_size));

  // Allocate flag.
  flag_ = at::ones(
      {NUM_BLOCKS},
      at::dtype(at::ScalarType::Long).device(at::kCUDA).device_index(device));
  params_.barrier_flag_ptr = reinterpret_cast<uint64_t*>(flag_.data_ptr());
}

MultiGpuBarrier::~MultiGpuBarrier() {
  for (size_t i = 0; i < params_.num_ranks; i++) {
    if (i == params_.rank) {
      continue;
    }
    C10_HIP_CHECK(hipIpcCloseMemHandle(params_.peer_barrier_ptrs[i]));
  }

  C10_HIP_CHECK(hipFree(params_.peer_barrier_ptrs[params_.rank]));
}

void MultiGpuBarrier::run() {
  const c10::DeviceGuard device_guard({c10::kCUDA, device_});

  auto stream = c10::hip::getCurrentHIPStream(device_).stream();

  hipLaunchKernelGGL((multi_gpu_barrier_kernel), dim3(NUM_BLOCKS),
                     dim3(BLOCK_SIZE), 0, stream, params_);
  C10_HIP_CHECK(hipGetLastError());
}

at::Tensor MultiGpuBarrier::bufferHandle() {
  const c10::DeviceGuard device_guard({c10::kCUDA, device_});

  hipIpcMemHandle_t barrier_handle;
  C10_HIP_CHECK(hipIpcGetMemHandle(&barrier_handle,
                                   params_.peer_barrier_ptrs[params_.rank]));

  auto tensor = at::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))},
                          at::dtype(at::kByte).device(c10::kCPU));
  auto data_ptr = reinterpret_cast<hipIpcMemHandle_t*>(tensor.data_ptr());
  memcpy(data_ptr, &barrier_handle, sizeof(hipIpcMemHandle_t));

  return tensor;
};

void MultiGpuBarrier::setPeerBufferHandles(
    const std::vector<at::Tensor>& handles) {
  const c10::DeviceGuard device_guard({c10::kCUDA, device_});

  assert(handles.size() == params_.num_ranks);
  for (size_t i = 0; i < params_.num_ranks; i++) {
    if (i == params_.rank) {
      continue;
    }

    auto data_ptr = reinterpret_cast<hipIpcMemHandle_t*>(handles[i].data_ptr());

    void* barrier_mem_ptr;
    C10_HIP_CHECK(hipIpcOpenMemHandle(&barrier_mem_ptr, *data_ptr,
                                      hipIpcMemLazyEnablePeerAccess));
    params_.peer_barrier_ptrs[i] = reinterpret_cast<uint64_t*>(barrier_mem_ptr);
  }
}

std::shared_ptr<MultiGpuBarrier> create_barrier(int64_t device, int64_t rank,
                                                int64_t num_ranks) {
  return std::make_shared<MultiGpuBarrier>(device, rank, num_ranks);
}

}  // namespace amd_repro
