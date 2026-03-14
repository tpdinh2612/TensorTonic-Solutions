def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Phase 3: Post-training (Clamp at final_lr)
    if step >= total_steps:
        return float(final_lr)

    # Phase 1: Warmup
    if step < warmup_steps:
        # Avoid division by zero; if step < warmup_steps, warmup_steps is guaranteed > 0
        return float((step / warmup_steps) * initial_lr)

    # Phase 2: Linear Decay
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        return float(final_lr)
        
    progress = (step - warmup_steps) / decay_steps
    lr = initial_lr - progress * (initial_lr - final_lr)
    
    return float(lr)