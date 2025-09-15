# Enhanced Architecture (Retired Inline Version)

The previous `enhanced_architecture.py` file mixed large blocks of Markdown and example pseudo/placeholder code inside a Python module which caused >100 syntax errors (backticks, fenced code blocks, numbered lists rendered as code, etc.).

This document preserves the intent: multi-service architecture sketch, hardware abstraction examples (RealSense, Stereo USB, Structured Light) and next‑steps checklist. The executable code has been removed from the Python module for repository health.

If you plan to re‑implement this design:
- Create dedicated packages under `intraoral_scanner_prototype_v2/` (e.g. `services/`, `hardware/` already exist) instead of a monolithic script.
- Keep narrative/design docs in `docs/`.
- Add incremental, testable slices (e.g. start with a single async service harness with mock handlers).

> This file is purely informational and has no runtime effect.
