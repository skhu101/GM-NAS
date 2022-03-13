import torch

class FancyPCA:

    def __init__(self, eig_vec = None, eig_val = None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [ -0.5675,  0.7192,  0.4009 ],
                [ -0.5808, -0.0045, -0.8140 ],
                [ -0.5836, -0.6948,  0.4203 ],
            ]).t()
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean = torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor
