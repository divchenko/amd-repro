import torch
import torch.distributed as dist
import time

dist.init_process_group(backend="nccl")
gloo_pg = dist.new_group(backend="gloo")
rank = dist.get_rank()
device = rank
torch.cuda.set_device(device)


a = torch.randn((10000, 10000), device="cuda")
b = torch.randn((10000, 10000), device="cuda")


if rank == 0:
    for _ in range(200):
        torch.matmul(a, b)
    ipc_event = torch.cuda.Event(interprocess=True)
    event = torch.cuda.Event()
    event.record()
    ipc_event.record()

event_list = [None]
if rank == 0:
    event_list[0] = ipc_event.ipc_handle()
    
dist.broadcast_object_list(event_list, src=0, group=gloo_pg)

if rank != 0:
    event = torch.cuda.Event.from_ipc_handle(device, event_list[0])

while not event.query():
    print(f"{rank} Waiting ...")
    time.sleep(1)
print(f"{rank} Done")

dist.barrier(group=gloo_pg)
dist.destroy_process_group()
