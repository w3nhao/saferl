from typing import Union, List, Optional, Callable
import torch
import pdb
import time
from configs.inference_config import InferenceConfig
from utils.metrics import calculate_safety_score

class GradientGuidance:
    """Gradient guidance calculator"""
    
    def __init__(
        self,
        scaler: torch.Tensor,
        w_obj: float = 0,
        w_safe: float = 0,
        guidance_scaler: float = 1.0,
        nt: int = 122,
        mode: str = "test",
        Q: float = 0.0,
        safety_threshold: float = 5.0,
        device: str = 'cuda',
    ):
        self.scaler = scaler
        self.w_obj = w_obj
        self.w_safe = w_safe
        self.guidance_scaler = guidance_scaler
        self.Q = Q
        self.safety_threshold = safety_threshold
        self.nt = nt

    def calculate_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        args: x: [B, channel, padded_time], scaled
        return: [B]
        """
        x = x[:, :, :self.nt] * self.scaler.to(x.device)
        objective = - x[:, -2, :].mean(-1)

        s = calculate_safety_score(x)
        safe_cost = torch.maximum(
            s + self.Q - self.safety_threshold,
            torch.zeros_like(s)
        ).mean(-1)

        return self.w_obj * objective + self.w_safe * safe_cost
    
    def get_weight(self, x: torch.Tensor) -> torch.Tensor:
        """
        args: x: [B, channel, padded_time], scaled
        return: [B]
        """
        loss = self.calculate_loss(x)
        return torch.exp(- loss * self.guidance_scaler)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            loss = self.calculate_loss(x) * self.guidance_scaler
            # the grad_x of loss.sum() is same as the grad_x of loss
            grad_x = torch.autograd.grad(loss, x, grad_outputs=torch.ones_like(loss))[0]

            return grad_x

def get_gradient_guidance(
    x: torch.Tensor,
    scaler: torch.Tensor,
    w_obj: float = 0,
    w_safe: float = 0,
    guidance_scaler: float = 1.0,
    device: str = 'cuda',
    nt: int = 122,
    Q: float = 0.0,
    safety_threshold: float = 5.0,
) -> Callable:
    """Build guidance gradient function"""
    return GradientGuidance(
        scaler=scaler,
        w_obj=w_obj,
        w_safe=w_safe,
        guidance_scaler=guidance_scaler,
        device=device,
        nt=nt,
        Q=Q,
        safety_threshold=safety_threshold,
    )(x)

def calculate_weight(x: torch.Tensor, scaler: torch.Tensor,
                     nt: int, Q: float, safety_threshold: float,
                     w_obj: float, w_safe: float, guidance_scaler: float) -> torch.Tensor:
    """
    args: 
        x: [B, channel, padded_time], scaled, GPU
        state_target: [B, state_dim, time], original, GPU
    return: [B]
    """
    x = x[:, :, :nt] * scaler.to(x.device)
    objective = - x[:, -2, :].mean(-1)

    s = calculate_safety_score(x)
    safe_cost = torch.maximum(
        s + Q - safety_threshold,
        torch.zeros_like(s)
    ).mean(-1)

    loss = w_obj * objective + w_safe * safe_cost
    weight = torch.exp(- loss * guidance_scaler)
    return weight

def normalize_weights(weights):
    '''
    Args:
        weights: torch.Tensor, [B]
    Returns:
        normalized_weights: torch.Tensor, [B]
    '''
    # if inf, replace with max
    if torch.isinf(weights).any():
        non_inf_mask = ~torch.isinf(weights)
        max_non_inf = weights[non_inf_mask].max()
        weights[torch.isinf(weights)] = max_non_inf

    if weights.sum() == 0:
        normalized_weights = torch.ones_like(weights)
    else:
        normalized_weights = weights.shape[0] * weights / weights.sum()

    return normalized_weights

