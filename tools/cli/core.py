"""Core CLI framework and utilities."""

import logging
import argparse
import sys
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
from enum import Enum

from .config import CommandConfig, CLIConfig, ArgumentConfig, ArgumentType

logger = logging.getLogger(__name__)


class Command(ABC):
    """Base class for CLI commands."""

    def __init__(self, config: CommandConfig):
        """Initialize command.

        Args:
            config: Command configuration
        """
        self.config = config
        self.parser: Optional[argparse.ArgumentParser] = None

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute command.

        Args:
            args: Parsed command arguments

        Returns:
            Execution result dictionary
        """
        pass

    def setup_parser(self, subparsers) -> argparse.ArgumentParser:
        """Setup argument parser for command.

        Args:
            subparsers: Subparsers from parent parser

        Returns:
            Command parser
        """
        self.parser = subparsers.add_parser(
            self.config.name,
            help=self.config.description,
            aliases=self.config.aliases,
        )

        for arg_config in self.config.arguments:
            self._add_argument(arg_config)

        self.parser.set_defaults(handler=self.execute)
        return self.parser

    def _add_argument(self, arg_config: ArgumentConfig):
        """Add argument to parser.

        Args:
            arg_config: Argument configuration
        """
        arg_name = arg_config.name
        if arg_config.short_name:
            names = [arg_config.short_name, f"--{arg_config.name}"]
        else:
            names = [f"--{arg_config.name}"]

        kwargs = {
            "help": arg_config.help,
            "default": arg_config.default,
            "required": arg_config.required,
        }

        if arg_config.dest:
            kwargs["dest"] = arg_config.dest

        if arg_config.arg_type == ArgumentType.STRING:
            kwargs["type"] = str
        elif arg_config.arg_type == ArgumentType.INTEGER:
            kwargs["type"] = int
        elif arg_config.arg_type == ArgumentType.FLOAT:
            kwargs["type"] = float
        elif arg_config.arg_type == ArgumentType.BOOLEAN:
            kwargs["action"] = "store_true"
        elif arg_config.arg_type == ArgumentType.LIST:
            kwargs["nargs"] = "+"
        elif arg_config.arg_type == ArgumentType.CHOICE:
            kwargs["choices"] = arg_config.choices

        self.parser.add_argument(*names, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert command to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.config.name,
            "description": self.config.description,
            "aliases": self.config.aliases,
            "arguments": [
                {
                    "name": arg.name,
                    "type": arg.arg_type.value,
                    "required": arg.required,
                    "help": arg.help,
                }
                for arg in self.config.arguments
            ],
        }


class ArgumentParserHelper:
    """Helper utilities for argparse."""

    @staticmethod
    def create_parser(
        app_name: str, description: str = "", version: str = ""
    ) -> argparse.ArgumentParser:
        """Create main argument parser.

        Args:
            app_name: Application name
            description: Application description
            version: Application version

        Returns:
            ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            prog=app_name, description=description, add_help=True
        )

        if version:
            parser.add_argument(
                "--version", action="version", version=f"%(prog)s {version}"
            )

        return parser

    @staticmethod
    def add_subparsers(
        parser: argparse.ArgumentParser,
        title: str = "commands",
        description: str = "available commands",
    ):
        """Add subparsers to parser.

        Args:
            parser: Parent parser
            title: Subparsers title
            description: Subparsers description

        Returns:
            Subparsers object
        """
        return parser.add_subparsers(
            title=title,
            description=description,
            help="command help",
            dest="command",
        )

    @staticmethod
    def parse_args(
        parser: argparse.ArgumentParser, args: Optional[List[str]] = None
    ) -> argparse.Namespace:
        """Parse command-line arguments.

        Args:
            parser: Argument parser
            args: Arguments to parse (None for sys.argv)

        Returns:
            Parsed arguments
        """
        return parser.parse_args(args)


class CommandRegistry:
    """Registry for managing CLI commands.

    Manages command registration, lookup, and execution.
    """

    def __init__(self):
        """Initialize command registry."""
        self.commands: Dict[str, Command] = {}
        self.aliases: Dict[str, str] = {}
        logger.info("CommandRegistry initialized")

    def register(self, command: Command):
        """Register command.

        Args:
            command: Command to register
        """
        self.commands[command.config.name] = command
        logger.info(f"Command registered: {command.config.name}")

        # Register aliases
        for alias in command.config.aliases:
            self.aliases[alias] = command.config.name

    def get(self, command_name: str) -> Optional[Command]:
        """Get command by name or alias.

        Args:
            command_name: Command name or alias

        Returns:
            Command or None
        """
        # Check if it's an alias
        if command_name in self.aliases:
            command_name = self.aliases[command_name]

        return self.commands.get(command_name)

    def list_commands(self) -> List[Dict[str, Any]]:
        """List all registered commands.

        Returns:
            List of command info dictionaries
        """
        return [cmd.to_dict() for cmd in self.commands.values()]

    def has_command(self, command_name: str) -> bool:
        """Check if command exists.

        Args:
            command_name: Command name or alias

        Returns:
            True if command exists
        """
        return command_name in self.commands or command_name in self.aliases


class CLIFramework:
    """Main CLI framework for LLM-Whisperer.

    Manages command registration, parsing, and execution.
    """

    def __init__(self, config: CLIConfig):
        """Initialize CLI framework.

        Args:
            config: CLI configuration
        """
        self.config = config
        self.registry = CommandRegistry()
        self.parser = ArgumentParserHelper.create_parser(
            config.app_name, config.description, config.version
        )
        self.subparsers = ArgumentParserHelper.add_subparsers(self.parser)
        logger.info(f"CLIFramework initialized: {config.app_name}")

    def register_command(self, command: Command):
        """Register command with framework.

        Args:
            command: Command to register
        """
        self.registry.register(command)
        command.setup_parser(self.subparsers)
        logger.debug(f"Command setup complete: {command.config.name}")

    def register_commands(self, commands: List[Command]):
        """Register multiple commands.

        Args:
            commands: List of commands
        """
        for command in commands:
            self.register_command(command)

    def get_command(self, command_name: str) -> Optional[Command]:
        """Get command from registry.

        Args:
            command_name: Command name

        Returns:
            Command or None
        """
        return self.registry.get(command_name)

    def list_commands(self) -> List[Dict[str, Any]]:
        """List all registered commands.

        Returns:
            List of command info
        """
        return self.registry.list_commands()

    def run(self, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run CLI with arguments.

        Args:
            args: Command-line arguments (None for sys.argv)

        Returns:
            Execution result
        """
        try:
            parsed_args = ArgumentParserHelper.parse_args(self.parser, args)

            if not hasattr(parsed_args, "handler"):
                logger.warning("No command specified")
                self.parser.print_help()
                return {"error": "No command specified"}

            logger.info(f"Executing command: {parsed_args.command}")
            result = parsed_args.handler(parsed_args)

            if result is None:
                result = {"status": "success"}

            logger.info(f"Command completed: {parsed_args.command}")
            return result

        except SystemExit:
            # Catch SystemExit from argparse
            return {"status": "exit"}
        except Exception as e:
            logger.error(f"CLI execution error: {str(e)}")
            return {"error": str(e)}

    def add_help_command(self):
        """Add help command."""
        help_config = CommandConfig(
            name="help",
            description="Show help information",
        )

        class HelpCommand(Command):
            def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
                commands = self.registry.list_commands()
                return {
                    "app": self.config.app_name,
                    "version": self.config.version,
                    "commands": commands,
                }

        help_cmd = HelpCommand(help_config)
        self.registry.register(help_cmd)
        help_cmd.setup_parser(self.subparsers)

    def print_help(self):
        """Print help information."""
        self.parser.print_help()

    def execute_command(self, command_name: str, **kwargs) -> Dict[str, Any]:
        """Execute command programmatically.

        Args:
            command_name: Command name
            **kwargs: Command arguments

        Returns:
            Execution result
        """
        command = self.get_command(command_name)
        if not command:
            logger.error(f"Command not found: {command_name}")
            return {"error": f"Command not found: {command_name}"}

        try:
            # Create namespace with kwargs
            args = argparse.Namespace(**kwargs)
            result = command.execute(args)
            return result or {"status": "success"}
        except Exception as e:
            logger.error(f"Command execution error: {str(e)}")
            return {"error": str(e)}
