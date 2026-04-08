# CLI Module

Command-line interface framework for LLM-Whisperer with extensible command system.

## Features

- **Command Framework**: Build modular CLI commands with automatic parsing
- **Argument Parser**: Easy argument definition with type support
- **Command Registry**: Manage and lookup commands
- **Help Generation**: Automatic help text generation
- **Command Aliases**: Support command aliases for convenience
- **Type Support**: String, integer, float, boolean, list, and choice arguments

## Components

### CLIFramework

Main framework for building CLI applications.

```python
from tools.cli import CLIFramework, CLIConfig

config = CLIConfig(
    app_name="my-app",
    version="1.0.0",
    description="My CLI application"
)

cli = CLIFramework(config)
```

### Command

Base class for implementing CLI commands.

```python
from tools.cli import Command, CommandConfig, ArgumentConfig, ArgumentType

class MyCommand(Command):
    def execute(self, args) -> Dict[str, Any]:
        return {"result": "success"}

command = MyCommand(CommandConfig(
    name="mycommand",
    description="Do something"
))

cli.register_command(command)
```

### CommandRegistry

Manages command registration and lookup.

Features:
- Register and retrieve commands
- Support command aliases
- List all available commands
- Check command existence

### ArgumentParserHelper

Utilities for argument parsing.

Methods:
- `create_parser()`: Create main parser
- `add_subparsers()`: Add subcommand parsers
- `parse_args()`: Parse command-line arguments

## Argument Types

### ArgumentType

- `STRING`: String argument
- `INTEGER`: Integer argument
- `FLOAT`: Floating-point argument
- `BOOLEAN`: Boolean flag
- `LIST`: Multiple values
- `CHOICE`: Choose from options

## Configuration

### CLIConfig

```python
@dataclass
class CLIConfig:
    app_name: str
    version: str = "1.0.0"
    description: str = ""
    commands: List[CommandConfig] = field(default_factory=list)
    default_command: Optional[str] = None
    enable_help: bool = True
    enable_version: bool = True
```

### CommandConfig

```python
@dataclass
class CommandConfig:
    name: str
    description: str = ""
    handler: Optional[Callable] = None
    arguments: List[ArgumentConfig] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    hidden: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ArgumentConfig

```python
@dataclass
class ArgumentConfig:
    name: str
    arg_type: ArgumentType = ArgumentType.STRING
    required: bool = False
    default: Optional[Any] = None
    help: str = ""
    choices: Optional[List[Any]] = None
    short_name: Optional[str] = None
    dest: Optional[str] = None
```

## Examples

### Basic Command

```python
from tools.cli import CLIFramework, CLIConfig, Command, CommandConfig

class GreetCommand(Command):
    def execute(self, args):
        name = getattr(args, 'name', 'World')
        return {"message": f"Hello, {name}!"}

config = CLIConfig(app_name="greeter", version="1.0.0")
cli = CLIFramework(config)

command = GreetCommand(CommandConfig(
    name="greet",
    description="Greet someone"
))

cli.register_command(command)
cli.run(["greet", "--name", "Alice"])
```

### Command with Arguments

```python
from tools.cli import (
    CLIFramework, Command, CommandConfig, 
    ArgumentConfig, ArgumentType
)

class ProcessCommand(Command):
    def execute(self, args):
        input_file = args.input
        output_file = args.output
        verbose = args.verbose
        return {
            "input": input_file,
            "output": output_file,
            "verbose": verbose
        }

command = ProcessCommand(CommandConfig(
    name="process",
    description="Process files",
    arguments=[
        ArgumentConfig(
            name="input",
            arg_type=ArgumentType.STRING,
            required=True,
            help="Input file"
        ),
        ArgumentConfig(
            name="output",
            arg_type=ArgumentType.STRING,
            required=True,
            help="Output file"
        ),
        ArgumentConfig(
            name="verbose",
            arg_type=ArgumentType.BOOLEAN,
            short_name="-v",
            help="Verbose output"
        )
    ]
))

cli.register_command(command)
cli.run(["process", "--input", "in.txt", "--output", "out.txt", "-v"])
```

### Command with Choices

```python
from tools.cli import ArgumentConfig, ArgumentType

arg = ArgumentConfig(
    name="format",
    arg_type=ArgumentType.CHOICE,
    choices=["json", "csv", "xml"],
    required=True,
    help="Output format"
)
```

### Command Aliases

```python
command = GreetCommand(CommandConfig(
    name="greet",
    description="Greet someone",
    aliases=["hello", "hi"]
))

# Can now use: greet, hello, or hi
cli.run(["hello"])
cli.run(["hi"])
```

### Programmatic Execution

```python
result = cli.execute_command("process", 
    input="data.txt", 
    output="result.txt",
    verbose=True
)
```

## Usage

### Command Line

```bash
# Get help
python -m tools.cli help

# List commands
python -m tools.cli commands

# Run command
python -m tools.cli process --input data.txt --output out.txt
```

### Programmatic

```python
from tools.cli import CLIFramework, CLIConfig

config = CLIConfig(app_name="myapp")
cli = CLIFramework(config)

# Register commands
cli.register_commands([command1, command2])

# Run CLI
result = cli.run(["command1", "--option", "value"])

# List commands
commands = cli.list_commands()
```

## Error Handling

Commands can return error results:

```python
def execute(self, args):
    try:
        # Do something
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}
```

## License

MIT
