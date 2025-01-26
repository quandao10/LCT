import torch
import torch.distributed as dist

class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.saved_tensors[0]

        # grad_output_norm = torch.linalg.vector_norm(
        #     grad_output, dim=list(range(1, len(grad_output.shape))), keepdim=True
        # ).mean()
        grad_output_norm = torch.norm(grad_output).mean().item()
        # nccl over all nodes
        grad_output_norm = avg_scalar_over_nodes(
            grad_output_norm, device=grad_output.device
        )

        grad_output_normalized = weight * grad_output / (grad_output_norm + 1e-8)

        return grad_output_normalized, None

@torch.no_grad()
def avg_scalar_over_nodes(value: float, device):
    value = torch.tensor(value, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value.item()

def gradnorm(x, weight=1.0):
    weight = torch.tensor(weight, device=x.device)
    return GradNormFunction.apply(x, weight)