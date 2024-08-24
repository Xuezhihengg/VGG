import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# TODO
def load_pretrained_state_dict() -> None:
    pass

def load_resume_state_dict() ->None:
    pass

def accuracy(output: Tensor, target: Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    a, pred = output.topk(maxk,1)  # pred size:(b x maxk)
    pred = pred.t()  # (maxk x b)
    correct = pred.eq(target.view(1,-1).expand_as(pred))  # (b x 1) -> (1 x b) -> (5 x b)

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0,keepdim = True)
        results.append((correct_k.mul_(100.0 / batch_size)))
    return results

def param_avg_logger(
        model:nn.Module
):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            avg_grad = torch.mean(parameter.grad)
            print(f'\t{name} - grad_avg: {avg_grad}')
        if parameter.data is not None:
            avg_weight = torch.mean(parameter.data)
            print(f'\t{name} - param_avg: {avg_weight}')

