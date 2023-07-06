import torch
import torch.nn.functional as F

def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)

def log_add_exp(a, b):
    """Computes the logarithm of the sum of exponentials of two numbers in a numerically stable way."""
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

@torch.jit.script
def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m

@torch.jit.script
def sliced_logsumexp(x, slices):
    lse = torch.logcumsumexp(
        torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float('inf')),
        dim=-1)

    slice_starts = slices[:-1]
    slice_ends = slices[1:]

    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    slice_lse_repeated = torch.repeat_interleave(
        slice_lse,
        slice_ends - slice_starts, 
        dim=-1
    )
    return slice_lse_repeated

def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))
 
    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot

def sum_except_batch(x, num_dims=1):
    """Sums all dimensions except the first"""
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_categorical(log_x_0, log_dist):
    return (log_x_0.exp() * log_dist).sum(dim=1)
