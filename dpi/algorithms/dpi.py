"""Deep Policy Iteration trainer."""

from __future__ import annotations

from typing import Optional

import torch

from .policy_eval import policy_evaluation_step
from .policy_improve import policy_improvement_step
from ..utils.training import DPIConfig, TrainingHistory


def _infer_device(module: torch.nn.Module) -> torch.device:
    for p in module.parameters():
        return p.device
    return torch.device("cpu")


def _infer_dtype(module: torch.nn.Module) -> torch.dtype:
    for p in module.parameters():
        return p.dtype
    return torch.float32


class DPITrainer:
    def __init__(
        self,
        model,
        value_network: torch.nn.Module,
        policy_network: Optional[torch.nn.Module],
        config: DPIConfig,
    ) -> None:
        self.model = model
        self.value_net = value_network
        self.policy_net = policy_network
        self.config = config

        self.device = _infer_device(self.value_net)
        self.dtype = _infer_dtype(self.value_net)

        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=config.value_lr)
        self.policy_opt = None
        if self.policy_net is not None:
            self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=config.policy_lr)

        self.value_sched = None
        self.policy_sched = None
        if config.lr_decay and config.lr_decay < 1.0:
            self.value_sched = torch.optim.lr_scheduler.StepLR(
                self.value_opt, step_size=config.lr_decay_steps, gamma=config.lr_decay
            )
            if self.policy_opt is not None:
                self.policy_sched = torch.optim.lr_scheduler.StepLR(
                    self.policy_opt, step_size=config.lr_decay_steps, gamma=config.lr_decay
                )

    def train(self, n_iterations: int) -> TrainingHistory:
        history = TrainingHistory()

        for _ in range(n_iterations):
            s_batch = self.model.sample_states(self.config.batch_size)
            s_batch = s_batch.to(device=self.device, dtype=self.dtype)

            policy_loss = None
            if self.policy_net is not None and self.policy_opt is not None:
                for _ in range(self.config.n_improve_steps):
                    policy_loss = policy_improvement_step(
                        self.model,
                        s_batch,
                        self.policy_net,
                        self.value_net,
                        self.policy_opt,
                        ito_method=self.config.ito_method,
                        fd_step_size=self.config.fd_step_size,
                        fd_stencil=self.config.fd_stencil,
                    )

            value_loss = None
            for _ in range(self.config.n_eval_steps):
                value_loss = policy_evaluation_step(
                    self.model,
                    s_batch,
                    self.value_net,
                    self.policy_net,
                    self.value_opt,
                    eval_method=self.config.eval_method,
                    delta_t=self.config.delta_t,
                    ito_method=self.config.ito_method,
                    fd_step_size=self.config.fd_step_size,
                    fd_stencil=self.config.fd_stencil,
                )

            history.value_loss.append(value_loss)
            history.policy_loss.append(policy_loss)

            if self.value_sched is not None:
                self.value_sched.step()
            if self.policy_sched is not None:
                self.policy_sched.step()

        return history
