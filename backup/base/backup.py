import torch

class BackupFn:
    def __init__(self, **kwargs):
        pass

    def __call__(self, tree, gamma=0.95) -> torch.Tensor:
        return self.backup(tree, gamma=gamma)

    def backup(self, tree, gamma=0.95) -> torch.Tensor:
        raise NotImplementedError("'backup' not implemented")