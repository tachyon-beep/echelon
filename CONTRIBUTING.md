# Contributing

Echelon welcomes contributions. Here's how to get started.

## Setup

```bash
uv python install 3.13
uv sync -p 3.13
```

## Running Tests

```bash
# All tests
PYTHONPATH=. uv run pytest tests

# Fast unit tests
PYTHONPATH=. uv run pytest tests/unit

# Integration tests (slower)
PYTHONPATH=. uv run pytest tests/integration
```

## Code Style

```bash
uv run ruff check .
```

- Line length: 110
- Type hints expected throughout
- No legacy code or backwards compatibility shims

## Pull Requests

- Keep changes small and focused
- Add tests for new functionality
- Run the test suite before submitting
- No backwards compatibility required — this is pre-release

## Bugs

Bug reports live in `docs/bugs/`. Create a markdown file describing the issue, including:
- What you expected
- What happened
- Steps to reproduce

## Questions

Open a GitHub issue or reach out via email: john [at] foundryside.dev
