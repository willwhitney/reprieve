import torch


def numpy_wrap_torch(repr_fn, device):
    def _helper(x):
        with torch.no_grad():
            x = torch.as_tensor(x).to(device)
            x = repr_fn(x)
            return x.cpu().numpy()
    return _helper
