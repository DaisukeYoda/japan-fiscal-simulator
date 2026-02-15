# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**jpfs (Japan Fiscal Simulator)** is a New Keynesian DSGE model implementation for simulating fiscal policy effects on the Japanese economy. It features a 5-sector model (households, firms, government, central bank, financial), a QZ-based Blanchard-Kahn/Klein solution method, MCP server integration for Claude Desktop, and a CLI interface.

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

- `nk_model.py`: 14方程式構造NKモデル（state 5, control 9）をBKで解く
- `model.py`: Extended 16-variable DSGE model that wraps `NewKeynesianModel` and adds fiscal/risk/tax blocks
- `steady_state.py`: Steady-state solver
- `simulation.py`: `ImpulseResponseSimulator` and `FiscalMultiplierCalculator` for policy analysis
- `solver.py` / `linear_solver.py`: QZ-based Blanchard-Kahn solution methods (`linear_solver.py` is compatibility wrapper)

### Bayesian Estimation (`src/japan_fiscal_simulator/estimation/`)

- `mcmc.py`: Random Walk Metropolis-Hastings sampler (4 chains, 100K draws default)
- `state_space.py`: State space representation builder for the DSGE model
- `kalman_filter.py`: Kalman filter for likelihood computation
- `priors.py`: Prior distribution definitions for structural parameters
- `parameter_mapping.py`: Maps between model parameters and estimation parameters
- `data_loader.py` / `data_fetcher.py`: Japanese economic data loading and fetching
- `diagnostics.py`: Convergence diagnostics (Gelman-Rubin statistics, etc.)
- `results.py`: Estimation results management

### Parameters (`src/japan_fiscal_simulator/parameters/`)

- `defaults.py`: Parameter dataclasses for each sector (`HouseholdParameters`, `FirmParameters`, `GovernmentParameters`, `CentralBankParameters`, `FinancialParameters`, `ShockParameters`)
- `calibration.py`: `JapanCalibration` class with Japan-specific parameter presets (low interest rate, high debt, 10% consumption tax)
- `constants.py`: Named constants for model coefficients

### Key Design Patterns

- **Dependency Injection**: `DSGEModel` receives `DefaultParameters` via constructor
- **Lazy Computation with Caching**: Properties like `steady_state` and `policy_function` compute on first access, cache results, and can be invalidated via `invalidate_cache()`
- **Immutable Parameter Objects**: All parameter dataclasses use `frozen=True`

### Variable System

The model tracks 16 variables (defined in `VARIABLE_INDICES`):
- `y`, `c`, `i`, `n`, `k`, `pi`, `r`, `R`, `w`, `mc`, `g`, `b`, `tau_c`, `a`, `q`, `rk`
- Shocks: `e_a`, `e_g`, `e_m`, `e_tau`, `e_risk`, `e_i`, `e_p`

## Python Version

Requires Python 3.14+. Type annotations use modern syntax without `from __future__ import annotations`.

## Coding Standards

This project follows Python best practices:
- Type hints on all functions and methods
- Dataclasses for data structures
- Protocol-based interfaces over inheritance

## PR and Commit guideline

PRは日本語で行う。コミット文は簡潔に書く（多くても2-3行）。


## Roadmap

@docs/ROADMAP_CENTRAL_BANK_LEVEL.md
