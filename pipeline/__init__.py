"""Independent video claim verification; no dataset or website writes."""

from .pipeline import verify_claim
from .batch import run_true_test

__all__ = ["verify_claim", "run_true_test"]
