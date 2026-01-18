import os
import torch

print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
print(f"RANK: {os.environ.get('RANK', 'NOT SET')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")

if torch.cuda.is_available():
    print(f"\nTotal GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if torch.distributed.is_initialized():
        print(f"\nDDP Rank: {torch.distributed.get_rank()}")
        print(f"DDP World Size: {torch.distributed.get_world_size()}")
        print(f"Device for rank: {torch.cuda.current_device()}")

