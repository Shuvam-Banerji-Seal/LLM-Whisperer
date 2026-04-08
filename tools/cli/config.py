"""Configuration dataclasses for CLI module."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum


class ArgumentType(Enum):
    """Types of CLI arguments."""

    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    LIST = "list"
    CHOICE = "choice"


@dataclass
class ArgumentConfig:
    """Configuration for a command argument.

    Attributes:
        name: Argument name
        arg_type: Argument type
        required: Whether argument is required
        default: Default value
        help: Help text
        choices: Valid choices for choice type
        short_name: Short flag name (e.g., -v)
        dest: Destination variable name
    """

    name: str
    arg_type: ArgumentType = ArgumentType.STRING
    required: bool = False
    default: Optional[Any] = None
    help: str = ""
    choices: Optional[List[Any]] = None
    short_name: Optional[str] = None
    dest: Optional[str] = None


@dataclass
class CommandConfig:
    """Configuration for a CLI command.

    Attributes:
        name: Command name
        description: Command description
        handler: Handler function
        arguments: List of argument configurations
        aliases: Command aliases
        hidden: Whether command is hidden from help
        metadata: Additional metadata
    """

    name: str
    description: str = ""
    handler: Optional[Callable] = None
    arguments: List[ArgumentConfig] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    hidden: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CLIConfig:
    """Configuration for CLI framework.

    Attributes:
        app_name: Application name
        version: Application version
        description: Application description
        commands: List of command configurations
        default_command: Default command to run
        enable_help: Enable help command
        enable_version: Enable version command
    """

    app_name: str
    version: str = "1.0.0"
    description: str = ""
    commands: List[CommandConfig] = field(default_factory=list)
    default_command: Optional[str] = None
    enable_help: bool = True
    enable_version: bool = True
