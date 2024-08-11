/*
 * Copyright (c) 2024, Fireworks AI.
 */
#include <cstddef>

namespace amd_repro {

long reserve_virtual_memory(int device, size_t size);
long allocate(int device, long ptr, size_t size);
void deallocate(long ptr, long allocHandle, int size);
void export_import(long handle);

}  // namespace amd_repro
