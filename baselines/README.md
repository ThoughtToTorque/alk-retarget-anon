# Baselines

Every method in this directory is an **independent implementation written for
this benchmark** — none of it is the original authors' code. The mark-based
baseline follows the interface published for MOKA (Liu et al., 2024) and the
constraint-writing baseline follows the pipeline published for ReKep (Huang et
al., 2024); every adaptation made to run them on this benchmark is stated in
the paper where the baseline is introduced, and each family includes
oracle-assisted arms that upper-bound what the interface can achieve here
(several of which outscore our own method — see the paper).

If you believe any part of a baseline deviates from the published method it
follows, please open an issue; the per-rollout records make any such check
concrete.
