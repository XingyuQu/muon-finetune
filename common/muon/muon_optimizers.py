import torch
import math
from torch.optim import Optimizer
from .muon_optim_helper import zeropower_via_newtonschulz5

# ================= Helper Functions =================
# Note: PolarExpress coefficients are integrated into newtonschulz5 via using_pe parameter
zeropower_backends = dict(
    newtonschulz5=zeropower_via_newtonschulz5,
)

# Debug counter for Muon NS calls (to verify Muon is being used in deepspeed)
_MUON_NS_CALL_COUNT = 0

# ================= Muon Optimizer Class =================

class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized optimizer.
    Supports backend: NewtonSchulz.
    """
    def __init__(
        self,
        params,
        lr=3e-4,
        momentum=0.95,
        nesterov=True,
        backend='newtonschulz5',
        backend_steps=5,
        weight_decay=0.01,
        adamw_params=None,
        adamw_lr=1e-4,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0.01,
        ns_using_pe=False,
        ns_dtype="bf16",  # Newton-Schulz computation dtype: bf16, fp32, fp64
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps,
            weight_decay=weight_decay,
            is_muon=True,
            ns_using_pe=ns_using_pe,
            ns_dtype=ns_dtype,
        )

        super().__init__(list(params), defaults)

        # AdamW params group
        if adamw_params is not None and len(list(adamw_params)) > 0:
            self.add_param_group({
                'params': list(adamw_params),
                'lr': adamw_lr,
                'betas': adamw_betas,
                'eps': adamw_eps,
                'weight_decay': adamw_wd,
                'is_muon': False
            })


    @torch.no_grad()
    def step(self, closure=None):
        """Apply one Muon update (NS-orthogonalized momentum) to Muon groups, AdamW to the rest."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # ================== Muon Update ==================
            if group.get('is_muon', False):
                lr = group['lr']
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                nesterov = group['nesterov']
                
                # Select Backend
                backend_name = group.get('backend', 'newtonschulz5')
                backend_steps = group.get('backend_steps', 5)
                zeropower_backend = zeropower_backends[backend_name]

                for p in group['params']:
                    if p.grad is None:
                        continue

                    g = p.grad

                    # Reshape to 2D if needed
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    elif g.ndim < 2:
                        raise NotImplementedError("Muon requires at least 2D gradients")

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)

                    buf = state['momentum_buffer']

                    # Update momentum
                    buf.mul_(momentum).add_(g)

                    if nesterov:
                        g_update = g.add(buf, alpha=momentum)
                    else:
                        g_update = buf.clone()

                    # Apply Backend Orthogonalization
                    ns_dtype = group.get("ns_dtype")

                    # DEBUG: Track Muon NS calls to verify optimizer is being used
                    global _MUON_NS_CALL_COUNT
                    _MUON_NS_CALL_COUNT += 1
                    if _MUON_NS_CALL_COUNT <= 5:
                        print(f"[MUON DEBUG] NS call #{_MUON_NS_CALL_COUNT}: backend={backend_name}, "
                              f"shape={g.shape}, "
                              f"using_pe={group['ns_using_pe']}, ns_dtype={ns_dtype}", flush=True)

                    g_update = zeropower_backend(g_update, steps=backend_steps,
                                                 using_pe=group["ns_using_pe"],
                                                 ns_dtype=ns_dtype)

                    # Calculate scaling factor
                    scale_factor = 0.2 * math.sqrt(max(g.shape[0], g.shape[1]))

                    if weight_decay > 0:
                         p.data.mul_(1 - lr * weight_decay)

                    # Apply Muon update with Scaling
                    p.data.add_(g_update, alpha=-lr * scale_factor)

            # ================== AdamW Update ==================
            else:
                # AdamW logic remains identical to standard PyTorch implementation
                lr = group['lr'] # Scheduler updates this automatically!
                wd = group['weight_decay']
                beta1, beta2 = group['betas']
                eps = group['eps']

                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    state['step'] += 1
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    
                    denom = exp_avg_sq.sqrt().add_(eps)
                    
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                        
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
