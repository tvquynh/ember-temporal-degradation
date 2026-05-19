"""Project-wide seed convention.

Per memory rule F: 10 seeds standard for full runs, 2 seeds for prototyping/smoke.
"""

SEEDS = [42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]
PROTOTYPE_SEEDS = [42, 123]

NUM_SEEDS = len(SEEDS)
assert NUM_SEEDS == 10, "SEEDS must contain exactly 10 values"
