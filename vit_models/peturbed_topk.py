import torch
import torch.nn as nn


class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 500, sigma: float = 0.05):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x, current_sigma=0.05):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, current_sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 500, sigma: float = 0.05):
        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # f = r - a  # free inside reserved
        #
        # print(f'Before Top-K: total: {t/1e9:2.2f}, reserved: {r/1e9:2.2f}, '
        #       f'allocated: {a/1e9:2.2f}, free: {f/1e9:2.2f}')

        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)

        perturbed_x = x[:, None, :] + noise * sigma  # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices  # b, nS, k
        indices = torch.sort(indices, dim=-1).values  # b, nS, k

        # f_new = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        # perturbation_use = f - f_new
        # print(f'USED BY TOP-K PERTURBATIONS [GB]: {perturbation_use/1e9:2.2f}, '
        #       f'now left free [GB]: {f_new/1e9:2.2f}')
        # f = f_new

        # b, nS, k, d
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()

        # f_new = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        # one_hot_use = f - f_new
        # print(f'USED BY TOP-K ONE-HOT PERTURBATIONS [GB]: {one_hot_use/1e9:2.2f}, '
        #       f'now left free [GB]: {f_new/1e9:2.2f}')
        # f = f_new

        indicators = perturbed_output.mean(dim=1)  # b, k, d

        # f_new = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        # indicators_use = f - f_new
        # print(f'USED BY TOP-K INDICATORS [GB]: {indicators_use/1e9:2.2f}, '
        #       f'now left free [GB]: {f_new/1e9:2.2f}')
        # f = f_new
        # print(f'TOTAL TOP-K MEMORY USAGE [GB]: {(perturbation_use+one_hot_use+indicators_use)/1e9:2.2f}')

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                             / ctx.num_samples / ctx.sigma)
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None] * 5)