"""Problem 3 outer-coating optimization based on the Problem 1 mother model."""

from .config import Problem3Parameters
from .simulation import Problem3CaseResult, run_candidate
from .workflow import execute_workflow

__all__ = [
    "Problem3CaseResult",
    "Problem3Parameters",
    "execute_workflow",
    "run_candidate",
]

