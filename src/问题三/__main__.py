"""Command-line entry point for Problem 3."""

from .workflow import execute_workflow, print_terminal_summary


def main() -> None:
    results = execute_workflow()
    print_terminal_summary(results)


if __name__ == "__main__":
    main()

