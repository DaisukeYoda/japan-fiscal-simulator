---
title: Code Quality Analysis with pyscn
impact: HIGH
impactDescription: Automated code quality assessment
tags: [pyscn, analysis, complexity, dead-code, dependencies]
---

# Code Quality Analysis with pyscn [HIGH]

## Description
Use pyscn for structural code analysis. It detects dead code, duplicate code, circular dependencies, and excessive complexity. Built with Go + tree-sitter for extreme speed (100,000+ lines/sec), ideal for AI-assisted development workflows.

## Bad Example
```bash
# No automated quality analysis
# Issues accumulate silently:
# - Dead code after refactoring
# - Copy-pasted duplicate code
# - Circular dependencies
# - Functions with complexity > 20
```

## Good Example
```bash
# Install
uvx pyscn analyze .          # Run without installing
uv tool install pyscn        # Or install globally

# Full analysis with HTML report
pyscn analyze .

# JSON output for CI
pyscn analyze --json .

# Select specific analyses
pyscn analyze --select complexity .
pyscn analyze --select deps .
pyscn analyze --select deadcode .
pyscn analyze --select clones .
pyscn analyze --select complexity,deps,deadcode .

# CI quality gate (pass/fail)
pyscn check .
pyscn check --max-complexity 15 .
pyscn check --max-cycles 0 .          # Enforce acyclic dependencies
pyscn check --allow-circular-deps .   # Warn instead of fail

# Generate config file
pyscn init
```

```toml
# .pyscn.toml or [tool.pyscn] in pyproject.toml
[complexity]
max_complexity = 15

[dead_code]
min_severity = "warning"

[output]
directory = "reports"
```

## Analysis Types

| Analysis | Detects |
|----------|---------|
| `complexity` | Cyclomatic complexity > threshold |
| `deadcode` | Unreachable code after exhaustive conditionals |
| `clones` | Duplicate code (Types 1-4) for refactoring |
| `deps` | Circular dependencies, high coupling (CBO) |

## MCP Integration

```bash
# Claude Code
claude mcp add pyscn -- uvx pyscn-mcp

# Use naturally in conversation:
# "Analyze code quality of src/"
# "Find duplicate code and help refactor"
# "Show complex functions and simplify them"
```

## Notes
- Use `pyscn check` in CI for quality gates
- Clone detection uses LSH acceleration for large codebases
- Complexity threshold of 10-15 is recommended
- Run after major refactoring to catch dead code

## References
- [pyscn GitHub](https://github.com/ludo-technologies/pyscn)
