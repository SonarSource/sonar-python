from typing import Any, IO, Sequence, Set, Tuple
import argparse
from django.core.management.base import SystemCheckError

class BaseCommand:
    # Metadata about this command.
    help: str

    # Configuration shortcuts that alter various logic.
    _called_from_command_line: bool
    output_transaction: bool  # Whether to wrap the output in a "BEGIN; COMMIT;"
    requires_migrations_checks: bool
    requires_system_checks: str | Sequence[str]
    # Arguments, common to all commands, which aren't defined by the argument parser.
    base_stealth_options: Tuple[str, ...]
    # Command-specific options not defined by the argument parser.
    stealth_options: Tuple[str, ...]
    suppressed_base_arguments: Set[str]

    stdout: IO[str]
    stderr: IO[str]

    def __init__(
        self,
        stdout: IO[str] | None = None,
        stderr: IO[str] | None = None,
        no_color: bool = False,
        force_color: bool = False,
    ) -> None: ...
    def create_parser(
        self,
        prog_name: str,
        subcommand: str | None,
    ) -> argparse.ArgumentParser: ...
    def add_arguments(self, parser: argparse.ArgumentParser) -> None: ...
    def handle(self, *args: Any, **options: Any) -> str | None: ...
    def execute(self, *args: Any, **options: Any) -> str | None: ...
    def print_help(self, prog_name: str, subcommand: str | None) -> None: ...
    def get_version(self) -> str: ...
