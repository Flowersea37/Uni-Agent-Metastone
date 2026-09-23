# test_nccl.py
import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")

rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)

print(
    "rank =", rank,
    "local_rank =", local_rank,
    "host =", os.uname().nodename,
    "NCCL_SOCKET_IFNAME =", os.getenv("NCCL_SOCKET_IFNAME"),
    flush=True,
)

x = torch.ones(768, device="cuda")

for i in range(100):
    dist.broadcast(x, src=0)
    torch.cuda.synchronize()

    if rank == 0:
        print("broadcast", i, "OK", flush=True)

dist.destroy_process_group()