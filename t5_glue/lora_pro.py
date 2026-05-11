"""Implementation of LoRA-Pro optimizer (AdamW variant).
Adapted for PEFT-style LoRA parameter names (lora_A/lora_B).
"""
from typing import cast, Tuple, Union, List
import math

import torch
from torch.optim import Optimizer
from torch.optim.adamw import adamw


def _get_scalar_dtype() -> torch.dtype:
    return torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32


def _match_lora_name(name: str) -> Tuple[str, str]:
    """Match a parameter name against known LoRA-A/B suffixes; return (tag, 'weight_a'|'weight_b') or empty pair."""
    for tag, key in (
        ("lora_embedding_A", "weight_a"),
        ("lora_embedding_B", "weight_b"),
        ("lora_A", "weight_a"),
        ("lora_B", "weight_b"),
        ("weight_a", "weight_a"),
        ("weight_b", "weight_b"),
    ):
        if tag in name:
            return tag, key
    return "", ""


def solve_sylvester(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Solve the Sylvester equation A X - X B = C via eigendecomposition (promotes bf16 to fp32)."""
    if A.dtype is torch.bfloat16:
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        C = C.to(torch.float32)
    B = -B
    m = B.shape[-1]
    n = A.shape[-1]
    R, U = torch.linalg.eig(A)
    S, V = torch.linalg.eig(B)
    F = torch.linalg.solve(U, torch.matmul((C + 0j), V))
    W = R[..., :, None] - S[..., None, :]
    Y = F / W
    X = torch.matmul(
        torch.matmul(U[..., :n, :n], Y[..., :n, :m]),
        torch.linalg.inv(V)[..., :m, :m],
    )
    return (
        X.real
        if all(torch.isreal(x.flatten()[0]) for x in [A, B, C])
        else X
    )


class LoRAProAdamW(Optimizer):
    """LoRA-Pro AdamW variant: groups LoRA-A/B pairs and updates them via an equivalent-grad Sylvester step."""

    def __init__(
        self,
        named_params: List[Tuple[str, torch.Tensor]],
        lr: Union[float, torch.Tensor] = 1e-3,
        lora_scaler: float = 2.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
        X_mode: str = "sylvester",
        lora_plus_scaler: int = 1,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if X_mode not in ["zero", "sylvester", "symmetry"]:
            raise ValueError(
                f"Invalid mode value: {X_mode}, mode should be in ['zero', 'sylvester', 'symmetry']"
            )

        self.X_mode = X_mode
        self.step_ = 0
        self.lora_plus_scaler = lora_plus_scaler
        self.named_param_dtype = {}
        self.fake_step = torch.tensor(0.0, dtype=_get_scalar_dtype())

        if not isinstance(named_params, list):
            named_params = [named_params]

        params = []
        for named_params_group in named_params:
            group_lr = named_params_group.get("lr", lr)
            param_group = {
                "params": [],
                "params_fp32": [],
                "names": [],
                "lr": group_lr,
            }
            for name, param in named_params_group["params"]:
                param_group["params"].append(param)
                param_group["params_fp32"].append(param.detach().clone().float())
                param_group["names"].append(name)
                self.named_param_dtype[name] = param.dtype
            params.append(param_group)

        defaults = dict(
            lr=lr,
            lora_scaler=lora_scaler,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            differentiable=differentiable,
            X_mode=X_mode,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._update_group_params(group)
            for param_fp32, param in zip(group["params_fp32"], group["params"]):
                param.data.copy_(param_fp32.data)
        return loss

    def _update_group_params(self, group):
        beta1, beta2 = cast(Tuple[float, float], group["betas"])
        lora_scaler = group["lora_scaler"]

        param_dict, grad_dict = {}, {}
        for i in range(len(group["params"])):
            param = group["params"][i]
            param_fp32 = group["params_fp32"][i]
            grad = param.grad
            name = group["names"][i]
            if grad is None:
                continue
            lora_tag, lora_key = _match_lora_name(name)
            if lora_key:
                base_name = name[: name.find(lora_tag)]
                param_dict[lora_key] = param_fp32
                grad_dict[lora_key] = grad
                if len(param_dict.keys()) == 1:
                    continue
                if len(param_dict.keys()) == 2:
                    name = base_name + "lora"
            state = self.state[name]

            if len(state) == 0:
                self._initialize_state(state, param_dict, param_fp32, group)

            if len(param_dict.keys()) == 2:
                self._update_lora_params(state, param_dict, grad_dict, group, lora_scaler)
                param_dict = {}
                grad_dict = {}
            else:
                if group["amsgrad"]:
                    max_exp_avg_sqs = [state["max_exp_avg_sq"]]
                else:
                    max_exp_avg_sqs = []

                adamw(
                    params=[param_fp32],
                    grads=[grad.to(torch.float32)],
                    exp_avgs=[state["exp_avg"]],
                    exp_avg_sqs=[state["exp_avg_sq"]],
                    max_exp_avg_sqs=max_exp_avg_sqs,
                    state_steps=[state["step"]],
                    amsgrad=group["amsgrad"],
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                )

    def _initialize_state(self, state, param_dict, p, group):
        state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
        if len(param_dict.keys()) == 2:
            self._initialize_lora_state(state, param_dict, group["amsgrad"])
        else:
            self._initialize_standard_state(state, p, group["amsgrad"])

    def _initialize_lora_state(self, state, param_dict, amsgrad):
        A = param_dict["weight_a"]
        B = param_dict["weight_b"]
        eq_shape = (B.shape[0], A.shape[1])
        state["exp_avg_eq"] = torch.zeros(
            eq_shape, device=A.device, dtype=A.dtype
        )
        state["exp_avg_sq_eq"] = torch.zeros(
            eq_shape, device=A.device, dtype=A.dtype
        )

    def _initialize_standard_state(self, state, p, amsgrad):
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        if amsgrad:
            state["max_exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

    def _update_lora_params(self, state, param_dict, grad_dict, group, lora_scaler):
        A: torch.Tensor = param_dict["weight_a"]
        B: torch.Tensor = param_dict["weight_b"]
        lora_rank, _ = A.shape
        out_features, _ = B.shape
        grad_A_orin_fp32 = grad_dict["weight_a"].to(torch.float32)
        grad_B_orin_fp32 = grad_dict["weight_b"].to(torch.float32)

        beta1, beta2 = cast(Tuple[float, float], group["betas"])
        eps = group["eps"]
        delta = 1e-8
        grad_scale = 1 / (lora_scaler**2)

        AA_T = torch.matmul(A, A.T)
        B_TB = torch.matmul(B.T, B)
        AA_T_inv = torch.linalg.pinv(
            AA_T + delta * torch.eye(lora_rank, device=A.device)
        ).to(A.dtype)

        step = int(state["step"].item())
        if step == 0:
            grad_A = grad_A_orin_fp32
            grad_B = grad_scale * torch.matmul(grad_B_orin_fp32, AA_T_inv)
        else:
            B_TB_inv = torch.linalg.pinv(
                B_TB + delta * torch.eye(lora_rank, device=A.device)
            ).to(A.dtype)
            B_TB_inv_B_T = torch.matmul(B_TB_inv, B.T)
            I_minus_BBT_inv = torch.eye(out_features, device=B.device, dtype=B.dtype) - torch.matmul(
                B, B_TB_inv_B_T
            )
            grad_A = grad_scale * torch.matmul(B_TB_inv, grad_A_orin_fp32)
            grad_B = grad_scale * torch.matmul(
                I_minus_BBT_inv, torch.matmul(grad_B_orin_fp32, AA_T_inv)
            )

        equiv_grad = lora_scaler * (torch.matmul(B, grad_A) + torch.matmul(grad_B, A))

        exp_avg_eq: torch.Tensor = state["exp_avg_eq"]
        exp_avg_sq_eq: torch.Tensor = state["exp_avg_sq_eq"]
        exp_avg_eq.mul_(beta1).add_(equiv_grad, alpha=1 - beta1)
        exp_avg_sq_eq.mul_(beta2).addcmul_(equiv_grad, equiv_grad, value=1 - beta2)

        step = step + 1
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        denom = (exp_avg_sq_eq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        g = (exp_avg_eq / bias_correction1) / denom
        g = g.to(B.dtype)

        grad_A_orin_ = lora_scaler * torch.matmul(B.T, g)
        grad_B_orin_ = lora_scaler * torch.matmul(g, A.T)

        if step == 1:
            grad_A_final = grad_A_orin_
            grad_B_final = grad_scale * torch.matmul(grad_B_orin_, AA_T_inv)
        else:
            B_TB_inv = torch.linalg.pinv(
                B_TB + delta * torch.eye(lora_rank, device=A.device)
            ).to(A.dtype)
            B_TB_inv_B_T = torch.matmul(B_TB_inv, B.T)
            I_minus_BBT_inv = torch.eye(out_features, device=B.device, dtype=B.dtype) - torch.matmul(
                B, B_TB_inv_B_T
            )
            X = self._compute_X(
                group,
                B,
                A,
                lora_scaler,
                grad_A_orin_,
                grad_B_orin_,
                B_TB_inv,
                AA_T,
                B_TB,
            ).to(B.device).to(B.dtype)
            grad_A_final = grad_scale * torch.matmul(B_TB_inv, grad_A_orin_) + torch.matmul(
                X, A
            )
            grad_B_final = grad_scale * torch.matmul(
                I_minus_BBT_inv, torch.matmul(grad_B_orin_, AA_T_inv)
            ) - torch.matmul(B, X)

        if group["weight_decay"] != 0:
            A.add_(A, alpha=-group["lr"] * group["weight_decay"])
            B.add_(B, alpha=-group["lr"] * group["weight_decay"])
        A.add_(grad_A_final, alpha=-group["lr"])
        B.add_(grad_B_final, alpha=-group["lr"])

        state["step"].add_(1)

    def _compute_X(
        self,
        group,
        B,
        A,
        lora_scaler,
        grad_A_orin_fp32,
        grad_B_orin_fp32,
        B_TB_inv,
        AA_T,
        B_TB,
    ):
        if group["X_mode"] == "sylvester":
            return solve_sylvester(
                B_TB,
                AA_T,
                -(1 / lora_scaler**2)
                * torch.matmul(torch.matmul(B_TB_inv, grad_A_orin_fp32), A.T),
            )
        if group["X_mode"] == "symmetry":
            return -0.5 * (1 / lora_scaler**2) * torch.matmul(
                torch.matmul(B_TB_inv, B.T), torch.matmul(grad_B_orin_fp32, AA_T)
            )
        return torch.zeros((B_TB_inv.shape[0], B_TB_inv.shape[0]))
