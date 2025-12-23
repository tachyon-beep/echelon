# Risk Register

High-risk or high-complexity work items that require explicit risk reduction before starting.

**Rule:** No work package over "medium" risk/complexity without a risk reduction plan.

## Format

| ID | Risk | Impact | Likelihood | Mitigation | Status |
|----|------|--------|------------|------------|--------|
| R001 | Example risk | High | Medium | Mitigation approach | Open |

## Active Risks

| ID | Risk | Impact | Likelihood | Mitigation | Status |
|----|------|--------|------------|------------|--------|
| R001 | Unity integration complexity | High | High | Prototype socket protocol first; validate round-trip before full impl | Deferred |
| R002 | Hierarchical training credit assignment | High | Medium | Train bottom-up; freeze lower layers before training commanders | Planned |
| R003 | Multi-match batching performance | Medium | Medium | Benchmark single-match first; profile before parallelizing | Open |

## Deferred High-Risk Items

Items explicitly not started because risk reduction hasn't been done.

### Unity Headless Server (R001)

**Why deferred:** Full Unity integration is high-risk until socket protocol is validated.

**Risk reduction needed:**
- [ ] Define wire protocol (struct layout, endianness)
- [ ] Prototype Python socket client with mock server
- [ ] Validate round-trip latency meets SPS targets
- [ ] Test headless build on target Linux environment

### Hierarchical Command (R002)

**Why deferred:** Requires stable Soldier-level training first.

**Risk reduction needed:**
- [ ] Soldier model converges reliably
- [ ] Freezing/loading model weights works correctly
- [ ] Temporal abstraction wrapper tested
- [ ] Credit assignment approach validated on toy problem

## Resolved Risks

| ID | Risk | Resolution | Date |
|----|------|------------|------|
| - | - | - | - |

## Adding New Risks

When identifying high-risk work:
1. Add to Active Risks table
2. Document risk reduction steps needed
3. Don't start implementation until mitigation is in place
