"""Transactional soft threshold maintenance, bounded by a monotonic deadline."""
import time
import torch


def maintain(model, threshold, budget_seconds, validate, allowed_drop=0.0,
             protected=(), clock=time.monotonic):
    if threshold < 0 or budget_seconds <= 0 or allowed_drop < 0:
        raise ValueError("Invalid maintenance threshold/budget/allowed_drop")
    # Validation uses the same caller-supplied fixed validation set before/after.
    before = float(validate(model))
    if not torch.isfinite(torch.tensor(before)):
        raise ValueError("Nonfinite validation score")
    deadline = clock() + budget_seconds
    snapshot = {}
    changed = 0
    exhausted = False
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if clock() >= deadline:
                    exhausted = True
                    break
                if param.ndim < 2 or not param.requires_grad or any(name.startswith(prefix) for prefix in protected):
                    continue
                snapshot[name] = param.detach().clone()
                # Copy + shrink one tensor is atomic at the Python level; GPU work
                # cannot be preempted mid-kernel. Check the clock after synchronization.
                param.copy_(param.sign() * (param.abs() - threshold).clamp_min(0))
                if param.is_cuda:
                    torch.cuda.synchronize(param.device)
                changed += param.numel()
        if clock() >= deadline:
            exhausted = True
        after = before if exhausted else float(validate(model))
        exhausted = exhausted or clock() >= deadline
        accepted = not exhausted and torch.isfinite(torch.tensor(after)).item() and after >= before - allowed_drop
    except Exception:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in snapshot:
                    param.copy_(snapshot[name])
        raise
    if not accepted:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in snapshot:
                    param.copy_(snapshot[name])
    return {"accepted": accepted, "before": before, "after": after,
            "parameters_considered": changed, "budget_exhausted": exhausted}
