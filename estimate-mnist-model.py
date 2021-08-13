import argparse
import time
from typing import Iterable
import numpy as np
import torch
from torch import nn

from torchvision import datasets, transforms

import json, os, math
import py3nvml.py3nvml as nvml

import torch.optim as optim

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--layers", type=int, default=0, help="number of hidden layers")

    parser.add_argument("--logdir", type=str, default="./", help="directory to log cuda stats")

    args = parser.parse_args()
    return args


def _generate_mem_hook(mem, hook_type, exp):
    def hook(module, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        if type(module).__name__ == "Normalize":
            return

        mem_stats = torch.cuda.memory_stats(torch.cuda.current_device())
        handle = nvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        torch.cuda.synchronize()
        
        mem.append({
            'call_idx': call_idx,
            'layer_type': type(module).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'nvml_mem_total': info.total,
            'nvml_mem_free': info.free,
            'nvml_mem_used': info.used,
            'pytorch_mem_all': mem_stats["allocated_bytes.all.current"],
            'pytorch_mem_cached': mem_stats["reserved_bytes.all.current"],
            'pytorch_peak_allocated': mem_stats["allocated_bytes.all.peak"]
        })

        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    return hook

def _add_memory_hooks(mem_log, exp, hr):
    h = nn.modules.module.register_module_forward_pre_hook(_generate_mem_hook(mem_log, 'pre', exp))
    hr.append(h)

    h = nn.modules.module.register_module_forward_hook(_generate_mem_hook(mem_log, 'fwd', exp))
    hr.append(h)

    # The following is deprecated but I do not know the difference between the two
    # h = nn.modules.module.register_module_backward_hook(_generate_mem_hook(mem_log, 'bwd', exp))

    # I do not understand why these different hook mechanisms generate different memory usages
    h = nn.modules.module.register_module_full_backward_hook(_generate_mem_hook(mem_log, 'bwd', exp))
    hr.append(h)

def actual(model, loss_fn, optimiser, train_loader):
    mem_log = []
    exp = "cost"
    hr = []

    _add_memory_hooks(mem_log, exp, hr)

    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        print((output.size(), output.element_size()))

        loss = loss_fn(output, target)
        if idx % 5 == 0:
            print(f"Loss: {loss.item()}")

        print((loss.size(), loss.element_size()))

        optimiser.zero_grad()
        
        loss.backward()
        
        optimiser.step()

        del data, target, output, loss

        if idx > 10:
            break

    for h in hr:
        h.remove()

    return mem_log

# This function is trying to peak estimate memory used by the live allocations at peak allocation.
# This assumes that memory usage isn't particularly optimsed
#   - The intial cost of the model (weights, params, etc.), optimisers, ... is a consistent baseline
#   - Input data and target, and outputs are deallocated after the training step
#   - Intermdiates allocated during a forward are deallocated after the corresponding backward
def estimate(model, loss_fn, optimiser, d_in, batchsize):
    with torch.no_grad():
        # Size of parameters (weights and biases - these show up as individual parameters for the
        # model but as groups for each module)
        # Two lots of memory are allocated for each: 
        #  1) For the values
        #  2) For buffers for gradients
        model_size = 0
        for param in model.parameters():
            model_size += np.prod(param.size()) * param.element_size()

        
        # if SDG
        # SDG optimiser doesn't use extra state
        # Account for the gradient buffers
        # model_size *=2

        # if Adam
        # Gradient buffers
        # Adam keeps two extra buffers for momentum
        model_size *= 4

        # Size of an input batch
        data = torch.randn(batchsize, d_in)
        model_size += np.prod(data.size()) * data.element_size()

        # TODO: Size of the input target? is this a thing
        # There seems to be some memory allocated beyond the input data
        # before the first forward step but only happens with the train_loader
        model_size += batchsize * 8

        # Account for size of intermediate outputs between layers on a forward
        # pass
        for module in model:
            data = module(data)
            model_size += (np.prod(data.size()) * data.element_size())

        # TODO: To figure out the general case
        # Memory is allocated for the actual classification in the forward pass?
        # e.g. batchsize many int64s
        model_size += batchsize * 8

        # Normalising, the loss function and optimisers also allocating memory - how much?
        # when are they deallocated?

        return int(model_size)

def main():
    d_in = 784
    d_hidden = 512
    d_out = 10
    args = get_args()
    torch.cuda.set_device(args.local_rank)

    #dist.init_process_group(backend=args.backend)
    torch.manual_seed(args.seed)

    model = nn.Sequential(*[ nn.Linear(d_in, d_hidden), nn.ReLU() ] +\
            [ layer for _ in range(args.layers) for layer in [nn.Linear(d_hidden, d_hidden), nn.ReLU()] ] +\
            [ nn.Linear(d_hidden, d_out) ])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda t: t.view(-1)),
        ]
    )
    dataset = datasets.MNIST("./mnist", download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 32)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    expected = estimate(model, loss_fn, optimiser, d_in, 32)

    nvml.nvmlInit()
    if torch.cuda.current_device() == 0:
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        mem_all, mem_cached = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        baseline = {
            'nvml_mem_total': info.total,
            'nvml_mem_free': info.free,
            'nvml_mem_used': info.used,
            'pytorch_mem_all': mem_all,
            'pytorch_mem_cached': mem_cached,
        }

        cost_log = actual(model.cuda(), loss_fn, optimiser, train_loader)

        with open(f'logs/cost_log.json', 'w') as json_file:
            json.dump({ "expected": expected, "baseline" : baseline, "run": cost_log }, json_file, indent=2, separators=(", ", ": "), sort_keys=True)

    nvml.nvmlShutdown()

if __name__ == "__main__":
    main()