"""Command-line entry point for the complete Problem 1 workflow."""

from .workflow import execute_workflow, print_terminal_summary


def main() -> None:
    """Execute all required simulations, CSV exports and terminal reporting."""

    results = execute_workflow()
    print_terminal_summary(results)


if __name__ == "__main__":
    main()
