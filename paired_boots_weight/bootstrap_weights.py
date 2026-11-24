"""Draw paired weights with a caller-supplied bootstrap sampler."""

import torch

def draw_paired_weights(batch_size, group_size, device, seed, iteration, sampler):
    """Call the original sampler with a temporary per-iteration CUDA seed."""
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("Paired training weights require CUDA")
    gpu_index = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[gpu_index]):
        with torch.cuda.device(gpu_index):
            torch.cuda.manual_seed(seed + iteration)
            return sampler(batch_size, group_size, dev)
