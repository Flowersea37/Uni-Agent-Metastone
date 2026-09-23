import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)

dist.init_process_group("nccl")

x = torch.tensor([rank + 1.0], device="cuda")

dist.all_reduce(x)

print(f"rank={rank}, result={x.item()}")

dist.destroy_process_group()