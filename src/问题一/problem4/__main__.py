"""Command-line entry point for the complete Problem 4 workflow."""

from .workflow import execute_workflow, print_terminal_summary


def main() -> None:
    results = execute_workflow()
    print_terminal_summary(results)


if __name__ == "__main__":
    main()
