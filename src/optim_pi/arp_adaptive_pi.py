import math, torch
from torch.optim import Optimizer

class ARPAdaptivePi(Optimizer):
    def __init__(self, params, *, lr=2e-3, alpha=0.01, mu=0.001,
                 pi_a=math.pi, G0=1.0, beta=5.0):
        defaults = dict(lr=lr, alpha=alpha, mu=mu,
                        pi_a=pi_a, G0=G0, beta=beta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, α, μ, πa, G0, β = (group[k] for k in
                                   ["lr", "alpha", "mu", "pi_a", "G0", "beta"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state.setdefault(p, {})
                if not st:
                    st["G"] = torch.zeros_like(p)
                    st["w_tilde"] = p.data.float().clone()

                G, w̃ = st["G"], st["w_tilde"]
                G += lr * (α * g.abs() - μ * G)                     # ARP update
                κ = πa / (math.pi * (1.0 + G / G0))                # curvature
                ϕ = torch.tanh(β * κ * g)                           # smooth sign
                w̃ -= lr * ϕ
                st["w_tilde"] = w̃
                ternary = torch.sign(w̃) * (w̃.abs() > 0.5)         # expose
                p.copy_(ternary)
        return loss
