# PyTorch Code Review: Batch 1

**Files Reviewed:**
- `/home/john/echelon/echelon/rl/model.py`
- `/home/john/echelon/scripts/train_ppo.py`
- `/home/john/echelon/scripts/eval_policy.py`

**Review Date:** 2025-12-23

---

## Summary

The codebase demonstrates solid PyTorch practices overall, with good device handling and LSTM state management. However, several critical bugs and optimization opportunities were identified:

**Critical Issues:**
1. Triple tensor conversion bug in train_ppo.py (lines 873-882)
2. Potential NaN risk in tanh inverse with inadequate clamping
3. Missing detach() in rollout collection causing gradient tracking overhead

**Performance Issues:**
1. Redundant tensor conversions (CPU↔GPU)
2. Inefficient advantage normalization pattern
3. Missing torch.backends.cudnn.benchmark flag

**Memory Leaks:**
1. No explicit cleanup of optimizer cache during device transfer
2. Potential CUDA cache accumulation without periodic emptying

---

## Bugs Found

### 1. CRITICAL: Triple Tensor Conversion (train_ppo.py:873-882)

**Location:** `/home/john/echelon/scripts/train_ppo.py:873-882`

**Issue:** The same tensor conversion is performed three times in a row:

```python
rewards_buf[step] = torch.from_numpy(rewards_flat).to(device)  # Line 873
next_done = torch.from_numpy(done_flat).to(device)             # Line 874
arena_done = torch.from_numpy(arena_done_np).to(opponent_device)  # Line 875
next_obs = torch.from_numpy(stack_obs_many(next_obs_dicts, blue_ids)).to(device)  # Line 876

rewards_buf[step] = torch.from_numpy(rewards_flat).to(device)  # Line 878 - DUPLICATE
next_done = torch.from_numpy(done_flat).to(device)             # Line 879 - DUPLICATE
next_obs = torch.from_numpy(stack_obs_many(next_obs_dicts, blue_ids)).to(device)  # Line 880 - DUPLICATE

next_obs = torch.from_numpy(stack_obs_many(next_obs_dicts, blue_ids)).to(device)  # Line 882 - DUPLICATE
```

**Impact:**
- Wasted computation (2x redundant tensor creation)
- Wasted memory allocations
- 3x calls to `stack_obs_many` when only 1 is needed
- Potential timing bugs if values change between assignments

**Fix:** Remove lines 878-882 entirely.

---

### 2. HIGH: Insufficient Clamping in _atanh (model.py:10)

**Location:** `/home/john/echelon/echelon/rl/model.py:10`

**Issue:** The clamping range `[-0.999, 0.999]` may not be sufficient to prevent NaN/Inf values when actions are very close to ±1.

```python
def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999, 0.999)  # Insufficient margin
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))
```

When `x = 0.999`:
- `log1p(0.999)` ≈ 0.693
- `log1p(-0.999)` ≈ -6.9
- Result ≈ 3.8 (OK)

When `x = 0.9999`:
- `log1p(-0.9999)` ≈ -9.2
- Result ≈ 5.0 (still OK)

When `x = 1.0 - 1e-7`:
- `log1p(-x)` → very large negative number
- Risk of numerical instability

**Impact:**
- Rare NaN/Inf values during training when actions saturate
- Training crashes or policy collapse
- Harder to debug because it's non-deterministic

**Recommendation:** Increase safety margin to `-0.9999` or add epsilon:

```python
def _atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))
```

---

### 3. MEDIUM: Log Probability Correction Stability (model.py:99)

**Location:** `/home/john/echelon/echelon/rl/model.py:99`

**Issue:** The log-determinant correction uses a fixed epsilon that may be too small:

```python
logprob = dist.log_prob(u).sum(-1) - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
```

**Analysis:**
- When `action → ±1`, `1.0 - action.pow(2) → 0`
- With `eps=1e-6`, `log(1e-6) ≈ -13.8`
- This can cause very large negative log probabilities
- PPO ratio calculations may become unstable

**Recommendation:** Use larger epsilon (1e-5 or 1e-4) and consider clamping the log term:

```python
logprob = dist.log_prob(u).sum(-1) - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6)).sum(-1)
```

---

### 4. MEDIUM: Missing detach() in Rollout Collection (train_ppo.py:753)

**Location:** `/home/john/echelon/scripts/train_ppo.py:753`

**Issue:** Action tensors are not detached before converting to numpy:

```python
action_np = action.cpu().numpy()  # Line 753
```

While the action was created inside a `torch.no_grad()` block (line 743), it's better practice to explicitly detach for clarity and to prevent accidental gradient tracking if the context changes.

**Impact:**
- Potential memory overhead if gradient tracking accidentally enabled
- Code fragility during refactoring

**Recommendation:**
```python
action_np = action.detach().cpu().numpy()
```

Similarly at line 82 in `eval_policy.py`:
```python
action_np = action_b.detach().cpu().numpy()  # Already has detach - GOOD
```

And at line 295 in `train_ppo.py` (evaluate_vs_heuristic):
```python
action_np = action_b.cpu().numpy()  # MISSING detach
```

---

### 5. LOW: Checkpoint Loading Without Strict Mode (train_ppo.py:592, 569)

**Location:** `/home/john/echelon/scripts/train_ppo.py:592, 569`

**Issue:** `load_state_dict` is called without `strict=True` parameter:

```python
model.load_state_dict(resume_ckpt["model_state"])  # Line 592
opp.load_state_dict(ckpt["model_state"])           # Line 569
```

**Impact:**
- Silent failures if checkpoint has missing or extra keys
- Version mismatch bugs harder to detect
- Model architecture changes may go unnoticed

**Recommendation:** Always use strict mode and handle mismatches explicitly:
```python
model.load_state_dict(resume_ckpt["model_state"], strict=True)
```

---

## Issues

### 1. Device Management: Redundant CPU↔GPU Transfers

**Locations:** Multiple

**Issue:** Several unnecessary round-trip transfers:

In `train_ppo.py:753`:
```python
action_np = action.cpu().numpy()  # GPU → CPU
# ... later ...
rewards_buf[step] = torch.from_numpy(rewards_flat).to(device)  # CPU → GPU (line 873)
next_obs = torch.from_numpy(stack_obs_many(...)).to(device)    # CPU → GPU (line 876)
```

**Optimization:** Could batch numpy operations and do single transfer, or keep more on GPU.

---

### 2. Memory: Inefficient Advantage Normalization (train_ppo.py:896)

**Location:** `/home/john/echelon/scripts/train_ppo.py:896`

**Issue:** Advantage normalization creates temporary tensors:

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Impact:**
- Extra memory allocations for mean/std computations
- Could be done in-place

**Recommendation:**
```python
adv_mean = advantages.mean()
adv_std = advantages.std()
advantages = (advantages - adv_mean) / (adv_std + 1e-8)
```

Or use in-place operations:
```python
advantages.sub_(advantages.mean()).div_(advantages.std() + 1e-8)
```

---

### 3. Initialization: Missing LSTM Weight Initialization (model.py:40)

**Location:** `/home/john/echelon/echelon/rl/model.py:40`

**Issue:** LSTM layers use default PyTorch initialization, but actor/critic heads have custom orthogonal initialization.

```python
self.lstm = nn.LSTM(self.hidden_dim, self.lstm_hidden_dim)  # Default init
# ... later ...
nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)      # Custom init
nn.init.orthogonal_(self.critic.weight, gain=1.0)           # Custom init
```

**Impact:**
- Inconsistent initialization strategy
- LSTM may have suboptimal initial weights
- Slower initial learning

**Recommendation:** Add explicit LSTM initialization:
```python
for name, param in self.lstm.named_parameters():
    if 'weight_ih' in name:
        nn.init.orthogonal_(param)
    elif 'weight_hh' in name:
        nn.init.orthogonal_(param)
    elif 'bias' in name:
        nn.init.zeros_(param)
```

---

### 4. Batch Handling: Potential Shape Mismatch Risk (train_ppo.py:601)

**Location:** `/home/john/echelon/scripts/train_ppo.py:601`

**Issue:** Observation stacking assumes fixed structure but doesn't validate shapes:

```python
next_obs = torch.from_numpy(stack_obs_many(next_obs_dicts, blue_ids)).to(device)
```

If `next_obs_dicts` is malformed or has wrong number of environments, this will fail silently or with cryptic error.

**Recommendation:** Add shape validation:
```python
obs_stacked = stack_obs_many(next_obs_dicts, blue_ids)
assert obs_stacked.shape == (batch_size, obs_dim), f"Shape mismatch: {obs_stacked.shape} vs ({batch_size}, {obs_dim})"
next_obs = torch.from_numpy(obs_stacked).to(device)
```

---

### 5. Memory Leak: Optimizer State Device Transfer (train_ppo.py:156-160)

**Location:** `/home/john/echelon/scripts/train_ppo.py:156-160`

**Issue:** `optimizer_to_device` moves optimizer state but doesn't clean up old device memory:

```python
def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
```

**Impact:**
- When resuming from checkpoint and switching devices, old device memory isn't freed
- GPU memory leak if moving CPU → GPU

**Recommendation:** Add explicit cleanup:
```python
def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                old_device = value.device
                state[key] = value.to(device)
                # If moving from GPU, free old memory
                if old_device.type == 'cuda' and device.type != old_device.type:
                    del value
                    torch.cuda.empty_cache()
```

---

### 6. Numerical Stability: GAE Computation (train_ppo.py:226-249)

**Location:** `/home/john/echelon/scripts/train_ppo.py:226-249`

**Issue:** GAE computation accumulates floating-point errors over long rollouts:

```python
for t in reversed(range(T)):
    # ...
    lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
    advantages[t] = lastgaelam
```

**Analysis:**
- With `gamma=0.99` and `gae_lambda=0.95`, the product `0.99 * 0.95 ≈ 0.9405`
- Over 512 steps (default rollout), this compounds: `0.9405^512 ≈ 1e-14`
- Negligible impact, but could use double precision for accumulation

**Recommendation:** If rollout_steps becomes very large (>1000), consider:
```python
lastgaelam = lastgaelam.double()  # Use FP64 for accumulation
# ... computation ...
advantages[t] = lastgaelam.float()  # Convert back
```

---

## Improvement Opportunities

### 1. Performance: Enable cuDNN Autotuner

**Location:** `/home/john/echelon/scripts/train_ppo.py` (missing)

**Opportunity:** Add cuDNN benchmarking for faster LSTM operations:

```python
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # or True if need reproducibility
```

**Impact:**
- 10-30% faster LSTM forward/backward passes
- Especially beneficial for fixed batch sizes

**Placement:** After line 465 (device setup).

---

### 2. Performance: Gradient Accumulation Opportunity

**Location:** `/home/john/echelon/scripts/train_ppo.py:924-927`

**Opportunity:** Current implementation does full backward pass per epoch. Could accumulate gradients for larger effective batch sizes:

```python
for epoch in range(args.update_epochs):
    # Current: full backward every epoch
    loss.backward()  # Line 925
```

**Recommendation:** Add gradient accumulation option for memory-constrained setups:
```python
parser.add_argument("--grad-accum-steps", type=int, default=1)

# In training loop:
for epoch in range(args.update_epochs):
    loss = loss / args.grad_accum_steps
    loss.backward()
    if (epoch + 1) % args.grad_accum_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

---

### 3. Memory: LSTM State Detachment

**Location:** `/home/john/echelon/echelon/rl/model.py:57-66`

**Opportunity:** LSTM states should be detached between rollout segments:

```python
def _step_lstm(self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor):
    # ...
    h = lstm_state.h * (1.0 - done_seq)  # No detach
    c = lstm_state.c * (1.0 - done_seq)
    y_seq, (h2, c2) = self.lstm(x_seq, (h, c))
    return y, LSTMState(h=h2, c=c2)  # Should detach?
```

**Current Behavior:** During rollout collection (with `no_grad`), this is OK. During training replay, gradients properly flow.

**Recommendation:** This is actually correct as-is for BPTT, but add comment:
```python
# Note: States are NOT detached to enable BPTT during training epochs
return y, LSTMState(h=h2, c=c2)
```

---

### 4. Robustness: Add NaN/Inf Checks

**Location:** `/home/john/echelon/scripts/train_ppo.py:922`

**Opportunity:** Add safety checks before optimizer step:

```python
loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

# Add checks:
if not torch.isfinite(loss):
    print(f"WARNING: Non-finite loss detected: {loss.item()}")
    print(f"  pg_loss={pg_loss.item()}, v_loss={v_loss.item()}, entropy={entropy_loss.item()}")
    # Option 1: Skip this update
    continue
    # Option 2: Reload from last checkpoint
    # Option 3: Halt training

optimizer.zero_grad(set_to_none=True)
loss.backward()
```

**Impact:**
- Catch numerical issues early
- Prevent corrupted model states
- Better debugging information

---

### 5. Performance: Pin Memory for DataLoader

**Location:** `/home/john/echelon/scripts/train_ppo.py:728-734`

**Opportunity:** Pre-allocated buffers could use pinned memory for faster transfers:

```python
obs_buf = torch.zeros(args.rollout_steps, batch_size, obs_dim, device=device)
```

**Recommendation:** If using CPU→GPU transfers, use pinned memory:
```python
if device.type == 'cuda':
    obs_buf = torch.zeros(args.rollout_steps, batch_size, obs_dim).pin_memory().to(device)
else:
    obs_buf = torch.zeros(args.rollout_steps, batch_size, obs_dim, device=device)
```

**Impact:**
- 20-30% faster CPU→GPU transfers
- Slight CPU memory overhead

---

### 6. Code Quality: Type Hints for Tensor Shapes

**Location:** Multiple

**Opportunity:** Add shape annotations using torchtyping or comments:

```python
# Current:
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    # ...
) -> tuple[torch.Tensor, torch.Tensor]:
```

**Recommendation:**
```python
def compute_gae(
    rewards: torch.Tensor,  # [T, B]
    values: torch.Tensor,   # [T, B]
    dones: torch.Tensor,    # [T, B]
    next_value: torch.Tensor,  # [B]
    next_done: torch.Tensor,   # [B]
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:  # ([T, B], [T, B])
```

Already present in some places (e.g., line 227-231) - good! Apply consistently.

---

### 7. Memory: Clear CUDA Cache Periodically

**Location:** `/home/john/echelon/scripts/train_ppo.py` (missing)

**Opportunity:** Add periodic CUDA cache clearing to prevent fragmentation:

```python
# After update loop (line 927):
if device.type == 'cuda' and update % 50 == 0:
    torch.cuda.empty_cache()
```

**Impact:**
- Prevents CUDA memory fragmentation over long training runs
- May slightly improve memory availability
- Minimal performance cost (only every 50 updates)

---

### 8. Checkpoint Safety: Atomic Saves

**Location:** `/home/john/echelon/scripts/train_ppo.py:1070, 1032`

**Opportunity:** Checkpoint saves are not atomic - if training crashes during save, checkpoint is corrupted:

```python
torch.save(ckpt, p)  # Line 1070
```

**Recommendation:** Use atomic saves via temporary file:
```python
import tempfile
import shutil

def save_checkpoint_atomic(state_dict: dict, path: Path) -> None:
    """Save checkpoint atomically to prevent corruption."""
    # Save to temporary file first
    tmp_path = path.with_suffix('.tmp')
    torch.save(state_dict, tmp_path)
    # Atomic rename (on POSIX systems)
    tmp_path.replace(path)

# Usage:
save_checkpoint_atomic(ckpt, p)
```

---

### 9. Device Management: Multi-GPU Support

**Location:** `/home/john/echelon/scripts/train_ppo.py:460-465`

**Opportunity:** Code has partial multi-GPU support (learning_device vs opponent_device) but doesn't use DataParallel or DistributedDataParallel:

```python
devices = resolve_devices(args.device)
device = devices[0] # Primary learning device
opponent_device = devices[1] if len(devices) > 1 else device
```

**Recommendation:** For scaling, add DDP support:
```python
if len(devices) > 1 and args.use_ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device
    )
```

**Impact:**
- Scales to multi-GPU training
- Requires refactoring for process groups

---

### 10. Logging: Add Gradient Norms Tracking

**Location:** `/home/john/echelon/scripts/train_ppo.py:926-927`

**Opportunity:** Track gradient norms for debugging:

```python
grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

# Log to metrics and W&B:
if update % 10 == 0:
    print(f"grad_norm={grad_norm:.3f}")

if wandb_run is not None:
    wandb_metrics["train/grad_norm"] = float(grad_norm)
```

**Impact:**
- Detect exploding/vanishing gradients
- Better debugging of training dynamics

---

## Recommendations Summary

### Immediate Action Required (Bugs):
1. **Fix triple tensor conversion** (train_ppo.py:873-882) - Remove duplicates
2. **Improve _atanh clamping** (model.py:10) - Increase safety margin to 1e-6
3. **Add detach() to rollout actions** (train_ppo.py:753, 295)

### High Priority (Issues):
4. **Enable cuDNN autotuner** for performance
5. **Add NaN/Inf checks** before optimizer steps
6. **Use atomic checkpoint saves** to prevent corruption
7. **Add strict=True to load_state_dict** calls

### Medium Priority (Improvements):
8. Initialize LSTM weights explicitly
9. Add gradient norm tracking
10. Add shape validation assertions
11. Optimize advantage normalization (in-place)

### Low Priority (Nice to Have):
12. Pin memory for faster transfers
13. Add periodic CUDA cache clearing
14. Add gradient accumulation option
15. Improve type hints with shape annotations

---

## Testing Recommendations

1. **Test checkpoint resume after crash**: Verify atomic saves work
2. **Test with saturating actions**: Ensure _atanh doesn't produce NaN
3. **Test multi-device training**: Verify device transfers are correct
4. **Memory profiling**: Run with `torch.cuda.memory_summary()` to check for leaks
5. **Gradient flow check**: Use `torch.autograd.grad_check` on model

---

## Overall Assessment

**Code Quality: 8/10**
- Well-structured, mostly following PyTorch best practices
- Good device management overall
- Proper use of no_grad contexts

**Correctness: 7/10**
- One critical bug (triple conversion)
- Some numerical stability concerns
- Missing edge case handling

**Performance: 7/10**
- Reasonable efficiency
- Some optimization opportunities remain
- Good use of vectorization

**Robustness: 6/10**
- Missing safety checks for NaN/Inf
- Non-atomic checkpoint saves
- Insufficient error handling for device mismatches
