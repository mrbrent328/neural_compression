"""Regression fix for E38 corpus count handling.

The corpus builder receives integer counts.  Keep those counters separate
from the generated lists; never call len() on the requested count.
"""

def normalize_count(value: int, minimum: int = 0) -> int:
    value = int(value)
    if value < minimum:
        raise ValueError(f"count must be >= {minimum}, got {value}")
    return value
