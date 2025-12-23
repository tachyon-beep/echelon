# Echelon Documentation

## Directory Structure

```
docs/
├── architecture/     # System design, component diagrams, data flow
├── bugs/             # Bug reports, triage, code review findings
├── plans/            # Implementation plans (executable task lists)
├── reference/        # Vision docs, specs, feature analysis
├── guides/           # How-to guides, onboarding, workflows
├── adr/              # Architecture Decision Records
└── archive/          # Completed or deprecated docs
    ├── implemented/  # Shipped features (historical reference)
    └── deferred/     # Paused or abandoned work
```

---

## Quick Navigation

### Active Work

| Priority | Document | Description |
|----------|----------|-------------|
| 1 | [bugs/TRIAGE.md](./bugs/TRIAGE.md) | Bug triage - 6 CRITICAL, 11 HIGH open |
| 2 | [plans/living-grid-nav-integration.md](./plans/living-grid-nav-integration.md) | Nav graph integration (ready to execute) |

### Reference

| Document | Description |
|----------|-------------|
| [reference/mech-tactics-drl-demo.md](./reference/mech-tactics-drl-demo.md) | Project vision, mechanics, roadmap |
| [reference/analysis.md](./reference/analysis.md) | Feature status and implementation notes |

### Code Reviews

| Document | Description |
|----------|-------------|
| [bugs/batch1_drl_review.md](./bugs/batch1_drl_review.md) | DRL review: training loop, PPO |
| [bugs/batch2_drl_review.md](./bugs/batch2_drl_review.md) | DRL review: environment, observations |
| [bugs/batch3_drl_review.md](./bugs/batch3_drl_review.md) | DRL review: arena, self-play |
| [bugs/batch4_drl_review.md](./bugs/batch4_drl_review.md) | DRL review: heuristic, rewards |

---

## Folder Purposes

### `architecture/`
System design documents describing how components fit together:
- Component diagrams
- Data flow diagrams
- API contracts between modules
- Performance architecture

### `bugs/`
Bug tracking and code quality:
- `TRIAGE.md` - Consolidated bug list with severity and status
- Code review outputs
- Regression reports

### `plans/`
Executable implementation plans following the writing-plans format:
- Bite-sized tasks with TDD steps
- Exact file paths and code
- Test commands and commit messages
- Use `superpowers:executing-plans` to implement

### `reference/`
Long-lived reference material:
- Project vision and goals
- Game mechanics specifications
- Feature analysis and status tracking
- External research notes

### `guides/`
How-to documentation for developers:
- Setup and installation
- Development workflows
- Testing procedures
- Deployment guides

### `adr/`
Architecture Decision Records (ADR):
- Document significant technical decisions
- Include context, decision, and consequences
- Format: `NNNN-title.md` (e.g., `0001-use-ppo-for-training.md`)

### `archive/`
Historical documents no longer actively maintained:
- `implemented/` - Shipped features (keep for reference)
- `deferred/` - Paused work (may resume later)

---

## Creating New Documents

### Implementation Plan
```bash
# Use date prefix for plans
docs/plans/YYYY-MM-DD-feature-name.md
```

### Architecture Decision Record
```bash
# Sequential numbering
docs/adr/NNNN-decision-title.md
```

### Bug Report
```bash
# Add to TRIAGE.md or create specific report
docs/bugs/YYYY-MM-DD-issue-description.md
```
