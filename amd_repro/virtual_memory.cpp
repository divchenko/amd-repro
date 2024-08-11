/*
 * Copyright (c) 2024, Fireworks AI.
 */
#include "virtual_memory.h"

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>

namespace amd_repro {

constexpr int64_t kMaxAllocPow2 = 30;

hipMemAllocationProp _get_alloc_prop(int device) {
  hipMemAllocationProp allocProp;
  allocProp.type = hipMemAllocationTypePinned;
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = device;
  allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  return allocProp;
}

void _map_allocated(int device,
                    const hipMemGenericAllocationHandle_t& allocHandle,
                    hipDeviceptr_t* ptr, size_t size) {
  C10_HIP_CHECK(hipMemMap(ptr, size, 0, allocHandle, 0));
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  C10_HIP_CHECK(hipMemSetAccess(ptr, size, &accessDesc, 1));
}

void export_import(long allocHandle) {
  int fd;
  C10_HIP_CHECK(hipMemExportToShareableHandle(
      &fd, reinterpret_cast<hipMemGenericAllocationHandle_t>(allocHandle),
      hipMemHandleTypePosixFileDescriptor, 0));

  hipMemGenericAllocationHandle_t importedHandle = {};
  C10_HIP_CHECK(hipMemImportFromShareableHandle(
      &importedHandle, (void*)fd, hipMemHandleTypePosixFileDescriptor));
}

long allocate(int device, long ptr, size_t size) {
  auto allocProp = _get_alloc_prop(device);
  hipMemGenericAllocationHandle_t allocHandle = {};
  C10_HIP_CHECK(hipMemCreate(&allocHandle, size, &allocProp, 0));
  _map_allocated(device, allocHandle, reinterpret_cast<hipDeviceptr_t*>(ptr),
                 size);
  return reinterpret_cast<long>(allocHandle);
}

void deallocate(long ptr, long allocHandle, int size) {
  C10_HIP_CHECK(hipMemUnmap(reinterpret_cast<hipDeviceptr_t*>(&ptr), size));
  C10_HIP_CHECK(hipMemRelease(
      reinterpret_cast<hipMemGenericAllocationHandle_t>(allocHandle)));
}

long reserve_virtual_memory(int device, size_t size) {
  at::globalContext().lazyInitCUDA();
  c10::DeviceGuard guard(at::Device(at::DeviceType::HIP, device));

  auto allocProp = _get_alloc_prop(device);
  size_t granularity = 0;
  C10_HIP_CHECK(hipMemGetAllocationGranularity(
      &granularity, &allocProp, hipMemAllocationGranularityRecommended));
  TORCH_CHECK_GT(granularity, 0);
  TORCH_CHECK(size % granularity == 0);

  char* ptr = nullptr;
  C10_HIP_CHECK(hipMemAddressReserve(reinterpret_cast<hipDeviceptr_t*>(&ptr),
                                     size, 0, 0, 0));
  return reinterpret_cast<long>(ptr);
}

}  // namespace amd_repro
