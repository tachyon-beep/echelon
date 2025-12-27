# Role-Specific Reward System Draft

**Date:** 2025-12-27
**Status:** DRAFT - For DRL Expert Review
**Context:** Scouts and Pack Leaders aren't learning to use their combat suite tools

---

## Problem Statement

Current training shows all roles learning similar behaviors (approach zone, fight). The specialized tools aren't being used:
- **Scouts** should paint targets for missile locks, not brawl
- **Pack Leaders** should issue orders and coordinate, not solo fight

The design doc (combat-suite-observation-architecture.md, Section 13.2) specifies role-differentiated rewards, but we only have `paint_assist_bonus` (delayed, indirect).

---

## Proposed Reward Components

### 1. Scout Rewards

#### 1.1 Paint Lock Reward (Immediate)
**Intent:** Reward scouts for successfully painting enemies, not just when teammates use the lock.

```python
# Trigger: Scout successfully paints an enemy (target in range, LOS, not already painted)
paint_lock_reward = 0.5  # Per successful paint application

# Conditions:
# - Painter is scout class
# - Target is enemy, alive, in painter range, has LOS
# - Target was not already painted by this pack
```

**Rationale:**
- Direct signal for the paint action
- 0.5 is small enough to not dominate, large enough to be learnable
- Stacks with existing `paint_assist_bonus` (2.0) when teammate uses the lock

#### 1.2 Paint Maintenance Reward (Per-Step)
**Intent:** Reward scouts for keeping paint locks active on valuable targets.

```python
# Trigger: Each step where scout has active paint lock on enemy
paint_maintenance_reward = 0.1 * target_value_mult  # Per step

# target_value_mult:
# - Heavy: 1.5 (high-value target)
# - Medium: 1.0
# - Light/Scout: 0.7 (lower priority)
```

**Rationale:**
- Encourages maintaining locks, not just spamming paint
- Higher reward for painting high-value targets (heavies)
- Creates continuous signal vs one-shot

#### 1.3 Scout Damage Penalty
**Intent:** Discourage scouts from brawling - their job is support.

```python
# Trigger: Scout deals direct damage (not paint-assisted)
scout_damage_penalty = -0.5 * damage_dealt / 100.0  # Scaled by damage

# Exceptions:
# - No penalty if scout is below 50% HP (self-defense)
# - No penalty if target was painted by scout (enabling team)
```

**Rationale:**
- Design doc says scouts should NOT be rewarded for damage
- Penalty pushes them toward support role
- Self-defense exception prevents learned helplessness

#### 1.4 Contact Reporting Reward (Future)
**Intent:** Reward scouts for discovering new enemies.

```python
# Trigger: Scout is first to detect an enemy (adds to team track store)
contact_report_reward = 0.3  # Per new contact discovered

# Requires: Track store implementation (not yet built)
```

**Status:** Deferred until track store system is implemented.

---

### 2. Pack Leader Rewards

#### 2.1 Order Issuance Reward (Immediate)
**Intent:** Encourage pack leaders to actually issue orders.

```python
# Trigger: Pack leader issues a valid order to a subordinate
order_issued_reward = 0.1  # Per order issued

# Conditions:
# - Order has valid target/objective
# - Assignee is alive and in pack
# - Cooldown: max 1 order reward per 5 seconds per subordinate
```

**Rationale:**
- Very small reward just to establish the behavior
- Cooldown prevents order spam
- The real value comes from order effectiveness

#### 2.2 Order Compliance Reward
**Intent:** Reward pack leaders when subordinates follow orders successfully.

```python
# Trigger: Subordinate completes assigned order objective
order_compliance_reward = 1.0 * order_difficulty_mult

# Order types and completion criteria:
# - FOCUS_FIRE: Target takes damage from assignee -> 0.5
# - FOCUS_FIRE: Target killed by assignee -> 2.0
# - ADVANCE: Assignee reaches objective area -> 1.0
# - HOLD: Assignee stays in position for duration -> 0.5
# - RALLY: Assignee reaches leader position -> 0.5
```

**Rationale:**
- Links leader reward to subordinate success
- Different order types have different completion criteria
- Creates hierarchical credit assignment

#### 2.3 Squad Survival Bonus
**Intent:** Pack leaders should keep their pack alive.

```python
# Trigger: End of episode
squad_survival_bonus = 0.5 * (alive_subordinates / total_subordinates)

# Only applies to pack leader role
```

**Rationale:**
- Encourages protective behavior
- Complements `team_reward_alpha` mixing
- End-of-episode to avoid micro-optimization

#### 2.4 Counterfactual Order Value (Advanced)
**Intent:** Reward orders that improve tactical position, not just outcomes.

```python
# From design doc Section 24.4:
def command_reward(order, before_state, after_state):
    ev_before = estimate_value(before_state, order.assignee_id)
    ev_after = estimate_value(after_state, order.assignee_id)
    return ev_after - ev_before
```

**Status:** Deferred - requires value function access and careful implementation.

---

### 3. Light Mech Adjustments

Since Light now has ECM/ECCM (moved from Scout):

#### 3.1 ECM Effectiveness Reward
**Intent:** Reward lights for using ECM to protect teammates.

```python
# Trigger: Friendly mech survives attack while in Light's ECM radius
ecm_protection_reward = 0.3  # Per "save"

# Definition of "save":
# - Enemy fires at friendly in ECM radius
# - Attack misses or deals reduced damage due to jamming
```

**Rationale:**
- ECM is now Light's unique tool
- Reward should encourage protective use
- Complements combat role (disrupt while fighting)

---

## Reward Weight Summary

| Component | Weight | Role | Frequency |
|-----------|--------|------|-----------|
| paint_lock | 0.5 | Scout | Per paint |
| paint_maintenance | 0.1 | Scout | Per step |
| paint_assist_bonus | 2.0 | Scout | Per teammate use |
| scout_damage_penalty | -0.5 | Scout | Per damage event |
| order_issued | 0.1 | Leader | Per order |
| order_compliance | 0.5-2.0 | Leader | Per completion |
| squad_survival | 0.5 | Leader | Per episode |
| ecm_protection | 0.3 | Light | Per save |

---

## Implementation Priority

**Phase 1 (Immediate):**
1. Scout paint_lock reward (immediate signal)
2. Scout damage penalty (role differentiation)

**Phase 2 (After observing Phase 1):**
3. Paint maintenance reward
4. Order issuance reward
5. Order compliance reward

**Phase 3 (Future):**
6. Contact reporting (needs track store)
7. Counterfactual order value (needs value function)
8. ECM protection reward (needs attack tracking)

---

## Open Questions for DRL Expert

1. **Credit assignment timing**: Should paint rewards be immediate or delayed by 1 step?
2. **Penalty magnitude**: Is -0.5 scout damage penalty too harsh? Could cause learned helplessness?
3. **Order spam prevention**: Is cooldown the right mechanism, or should we use diminishing returns?
4. **Curriculum**: Should we introduce these rewards gradually or all at once?
5. **PBRS compliance**: Do any of these violate potential-based reward shaping guarantees?

---

## Expected Behavior Changes

**Before:**
- Scouts brawl like other mechs
- Pack leaders fight solo
- Paint rarely used
- Orders never issued

**After:**
- Scouts hang back, maintain paint locks on heavies
- Pack leaders issue FOCUS_FIRE on painted targets
- Heavies fire missiles using paint locks
- Coordinated pack behavior emerges
