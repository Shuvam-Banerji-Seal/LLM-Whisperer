"""CLI module for LLM-Whisperer.

Provides command-line interface framework and utilities.
"""

from .core import (
    CLIFramework,
    CommandRegistry,
    Command,
    ArgumentParserHelper,
)
from .config import CommandConfig, CLIConfig, ArgumentConfig, ArgumentType

__all__ = [
    "CLIFramework",
    "CommandRegistry",
    "Command",
    "ArgumentParserHelper",
    "CommandConfig",
    "CLIConfig",
    "ArgumentConfig",
    "ArgumentType",
]
