import os
import torch
import torch.distributed as dist

dist.init_process_group("gloo")

rank = dist.get_rank()

x = torch.ones(768)

for i in range(1000):
    dist.broadcast(x, src=0)

    if rank == 0:
        print(f"broadcast {i} OK", flush=True)

dist.destroy_process_group()
