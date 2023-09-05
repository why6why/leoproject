from typing import Tuple, Any

import torch
from torch import nn


class BaseAttack(object):
    def __call__(self, ori_image: torch.Tensor, label: torch.Tensor):
        raise NotImplementedError


class BaseFGSMAttack(BaseAttack):
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.NLLLoss = nn.NLLLoss()
        self.NLLLoss.to(self.device)

    def fgsm_attack_preprocess(self, ori_image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _ori_image = ori_image.to(self.device)
        _label = label.to(self.device)
        _ori_image.requires_grad = True
        output = self.model(_ori_image)
        loss_init = self.NLLLoss(output, _label)
        self.model.zero_grad()
        loss_init.backward()
        data_grad = _ori_image.grad.data
        return _ori_image, data_grad

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PGDAttack(BaseAttack):
    def __init__(self, model: nn.Module, device: torch.device, eps=0.2, minbound=0.0, maxbound=1.0, iterations=15):
        self.model = model
        self.device = device
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.eps = eps
        self.minbound = minbound
        self.maxbound = maxbound
        self.iterations = iterations
        self.crossEntropyLoss.to(self.device)

    def __call__(self, ori_image: torch.Tensor, label: torch.Tensor):
        _ori_image = ori_image.to(self.device)
        _label = label.to(self.device)
        x = torch.clamp(_ori_image + torch.empty_like(_ori_image).uniform_(-self.eps, self.eps), self.minbound, self.maxbound).detach()
        for i in range(self.iterations):
            x.requires_grad = True
            logits = self.model(x)
            loss = -self.crossEntropyLoss(logits, _label)
            try:
                grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
            except:
                grad = 0
            x = x.detach() + (self.eps / self.iterations) * grad.sign()
            x = torch.clamp(x, self.minbound, self.maxbound)
        return x


class FGSMAttack(BaseFGSMAttack):
    def __init__(self, model: nn.Module, device: torch.device, epsilon=0.2):
        super().__init__(model, device)
        self.epsilon = epsilon

    def __call__(self, ori_image: torch.Tensor, label: torch.Tensor):
        _ori_image, data_grad = self.fgsm_attack_preprocess(ori_image, label)
        pert_out = _ori_image + self.epsilon * data_grad.sign()
        pert_out = torch.clamp(pert_out, 0, 1)
        return pert_out


class IFGSMAttack(BaseFGSMAttack):
    def __init__(self, model: nn.Module, device: torch.device, epsilon=0.2):
        super().__init__(model, device)
        self.epsilon = epsilon
        self.iter = 10

    def __call__(self, ori_image: torch.Tensor, label: torch.Tensor):
        _ori_image, data_grad = self.fgsm_attack_preprocess(ori_image, label)
        alpha = self.epsilon / self.iter
        pert_out = _ori_image
        for i in range(self.iter - 1):
            pert_out = pert_out + alpha * data_grad.sign()
            pert_out = torch.clamp(pert_out, 0, 1)
            if torch.norm((pert_out - _ori_image), p=float('inf')) > self.epsilon:
                break
        return pert_out


class MIFGSMAttack(BaseFGSMAttack):
    def __init__(self, model: nn.Module, device: torch.device, epsilon=0.2):
        super().__init__(model, device)
        self.epsilon = epsilon
        self.iter = 10
        self.decay_factor = 1.0
        self.g = 0
        self.alpha = self.epsilon / self.iter

    def __call__(self, ori_image: torch.Tensor, label: torch.Tensor):
        _ori_image, data_grad = self.fgsm_attack_preprocess(ori_image, label)
        pert_out = _ori_image

        for i in range(self.iter - 1):
            g = self.decay_factor * self.g + data_grad / torch.norm(data_grad, p=1)
            pert_out = pert_out + self.alpha * torch.sign(g)
            pert_out = torch.clamp(pert_out, 0, 1)  # 把值限制在0到1之间
            if torch.norm((pert_out - _ori_image), p=float('inf')) > self.epsilon:  # 无穷范数
                break
        return pert_out
