"""Command-line entry point for the complete Problem 2 workflow."""

from .workflow import execute_workflow, print_terminal_summary


def main() -> None:
    """Execute all Problem 2 simulations, checks, exports and reporting."""

    results = execute_workflow()
    print_terminal_summary(results)


if __name__ == "__main__":
    main()
