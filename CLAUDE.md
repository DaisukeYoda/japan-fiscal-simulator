# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**jpfs (Japan Fiscal Simulator)** is a New Keynesian DSGE model implementation for simulating fiscal policy effects on the Japanese economy. It features a 5-sector model (households, firms, government, central bank, financial), Blanchard-Kahn solution method, MCP server integration for Claude Desktop, and a CLI interface.

## Commands

```bash
# Install dependencies
uv sync

# Add packages (updates pyproject.toml + installs)
uv add <package>          # runtime dependency
uv add --dev <package>    # dev dependency

# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_model.py

# Run single test
uv run pytest tests/test_model.py::TestDSGEModel::test_steady_state_computation

# Type checking (strict mode)
uv run mypy src/japan_fiscal_simulator

# Lint and format
uv run ruff check src tests
uv run ruff format src tests

# CLI usage
uv run jpfs simulate consumption_tax --shock -0.02 --periods 40 --graph
uv run jpfs multiplier government_spending --horizon 8
uv run jpfs steady-state
uv run jpfs parameters
uv run jpfs mcp  # Start MCP server
```

## Architecture

### Core Model (`src/japan_fiscal_simulator/core/`)

- `nk_model.py`: Core 3-equation New Keynesian model (IS curve, Phillips curve, Taylor rule) with reduced-form solution
- `model.py`: Extended 14-variable DSGE model that wraps `NewKeynesianModel` and derives additional variables from steady-state relationships
- `steady_state.py`: Steady-state solver
- `simulation.py`: `ImpulseResponseSimulator` and `FiscalMultiplierCalculator` for policy analysis
- `solver.py` / `linear_solver.py`: Blanchard-Kahn solution methods

### Parameters (`src/japan_fiscal_simulator/parameters/`)

- `defaults.py`: Parameter dataclasses for each sector (`HouseholdParameters`, `FirmParameters`, `GovernmentParameters`, `CentralBankParameters`, `FinancialParameters`, `ShockParameters`)
- `calibration.py`: `JapanCalibration` class with Japan-specific parameter presets (low interest rate, high debt, 10% consumption tax)
- `constants.py`: Named constants for model coefficients

### Key Design Patterns

- **Dependency Injection**: `DSGEModel` receives `DefaultParameters` via constructor
- **Lazy Computation with Caching**: Properties like `steady_state` and `policy_function` compute on first access, cache results, and can be invalidated via `invalidate_cache()`
- **Immutable Parameter Objects**: All parameter dataclasses use `frozen=True`

### Variable System

The model tracks 14 variables (defined in `VARIABLE_INDICES`):
- State variables: `y` (output), `c` (consumption), `i` (investment), `n` (labor), `k` (capital), `pi` (inflation), `r` (real rate), `R` (nominal rate), `w` (wage), `mc` (marginal cost), `g` (government spending), `b` (debt), `tau_c` (consumption tax), `a` (technology)
- Shocks: `e_a`, `e_g`, `e_m`, `e_tau`, `e_risk`

## Python Version

Requires Python 3.14+. Type annotations use modern syntax without `from __future__ import annotations`.

## Coding Standards

This project follows Python best practices:
- Type hints on all functions and methods
- Dataclasses for data structures
- Protocol-based interfaces over inheritance
