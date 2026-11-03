"""Group-relative policy loss; only sampled assistant tokens receive loss."""


def group_advantages(rewards):
    import torch
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("A finite reward group requires at least two rollouts")
    # Preserve the units of correctness and cost. Per-case standardization
    # would amplify a tiny request-cost difference as much as a wrong answer.
    if values.max()-values.min() < 1e-6:
        return torch.zeros_like(values)
    return values-values.mean()


def grpo_token_loss(logps, old_logps, ref_logps, advantage, clip=.2, beta=.02):
    import torch
    if logps.shape != old_logps.shape or logps.shape != ref_logps.shape:
        raise ValueError("Policy, old-policy and reference tokens differ")
    ratio = (logps - old_logps.detach()).exp()
    objective = torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
    delta = ref_logps.detach() - logps
    kl = delta.exp() - delta - 1
    loss = -objective + beta * kl
    if not torch.isfinite(loss).all():
        raise ValueError("Nonfinite GRPO loss")
    return loss, kl


def trajectory_reward(session):
    """Trainer-only source-label/region reward, never a tool observation."""
    env = session._env
    evaluation = env.evaluate(env.decision) if env.finished else None
    correct = bool(evaluation and evaluation["grounded_correct"])
    reward = (1.0 if correct else -1.0) - .01*env.tool_cost - .02*env.request_cost - .02*session.errors
    return reward, correct
