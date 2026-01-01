"""stats — statistical-analysis harness (Phase 4).

Implements the frozen statistics protocol from baselines/docs/CODE_MAP.md:
  - N seeds shared across methods (paired design)
  - success rate + Wilson 95% CI
  - McNemar exact test (paired) / Fisher exact test (unpaired fallback)
  - Holm-Bonferroni multiple-comparison correction
  - alpha = 0.05, raw p-values always reported

Pure numpy + stdlib math; scipy is NOT required at runtime (it is only used
as a reference in the unit tests).  Python 3.8 compatible.

Modules:
  aggregate  — per-rollout JSON -> tidy table (list of dicts / CSV / pandas)
  tests      — Wilson CI, exact McNemar, Fisher exact, paired bootstrap, Holm
  power      — simulation-based power analysis / minimum-N justification
  report     — markdown + LaTeX (booktabs) tables, decoupling analysis
  selftest   — end-to-end check on existing phase-2 data -> demo_report.md
"""

__all__ = ["aggregate", "tests", "power", "report"]
