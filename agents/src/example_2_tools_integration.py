"""
Example 2: Tools Integration and Management

This example demonstrates:
- Creating tools from Python functions
- Managing tools with registry
- Tool validation and schemas
- Dynamic tool loading
- Tool discovery and search

Source: https://python.langchain.com/docs/how_to/custom_tools
"""

import sys
import os
import json
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_tools_integration import (
    FunctionTool,
    ToolRegistry,
    ToolCategory,
    ToolValidator,
    ToolSchema,
    ToolParameter,
    DynamicToolLoader,
    BaseTool,
)


def example_1_create_function_tool():
    """Create tools from Python functions."""
    print("=" * 60)
    print("Example 1: Creating Tools from Functions")
    print("=" * 60)

    # Define tool functions
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    def multiply_numbers(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    def string_length(text: str) -> int:
        """Get the length of a string."""
        return len(text)

    print("\nDefining tools:")
    print("  • add_numbers(a: int, b: int) -> int")
    print("  • multiply_numbers(a: float, b: float) -> float")
    print("  • string_length(text: str) -> int")

    # Create tools
    print("\nCreating tools with FunctionTool wrapper:")

    add_tool = FunctionTool(
        add_numbers, category=ToolCategory.COMPUTATION, tags=["math", "arithmetic"]
    )

    multiply_tool = FunctionTool(
        multiply_numbers, category=ToolCategory.COMPUTATION, tags=["math", "arithmetic"]
    )

    length_tool = FunctionTool(
        string_length, category=ToolCategory.UTILITY, tags=["string", "utility"]
    )

    print("✓ Tools created successfully")

    # Display tool schemas
    print("\nTool Schemas:")
    print("-" * 60)

    for tool in [add_tool, multiply_tool, length_tool]:
        schema = tool.get_schema()
        print(f"\nTool: {schema.name}")
        print(f"  Description: {schema.description}")
        print(f"  Category: {schema.category.value}")
        print(f"  Parameters:")
        for param in schema.parameters:
            print(f"    - {param.name} ({param.type}): {param.description}")
        print(f"  Return Type: {schema.return_type}")

    # Execute tools
    print("\n" + "-" * 60)
    print("Executing Tools:")
    print("-" * 60)

    try:
        result1 = add_tool.execute(a=10, b=5)
        print(f"\nadd_numbers(10, 5) = {result1}")

        result2 = multiply_tool.execute(a=3.5, b=2.0)
        print(f"multiply_numbers(3.5, 2.0) = {result2}")

        result3 = length_tool.execute(text="Hello, World!")
        print(f"string_length('Hello, World!') = {result3}")
    except Exception as e:
        print(f"Error executing tools: {e}")


def example_2_tool_registry():
    """Manage tools with a registry."""
    print("\n" + "=" * 60)
    print("Example 2: Tool Registry Management")
    print("=" * 60)

    # Create registry
    registry = ToolRegistry()
    print("\n✓ Tool registry created")

    # Define and register tools
    def search_documents(query: str) -> List[str]:
        """Search for documents matching the query."""
        return [f"Document about {query}"]

    def fetch_data(url: str) -> Dict:
        """Fetch data from a URL."""
        return {"url": url, "status": "fetched"}

    def parse_json(json_str: str) -> Dict:
        """Parse JSON string into dictionary."""
        return json.loads(json_str)

    search_tool = FunctionTool(
        search_documents,
        category=ToolCategory.DATA_RETRIEVAL,
        tags=["search", "documents"],
    )

    fetch_tool = FunctionTool(
        fetch_data, category=ToolCategory.NETWORK, tags=["network", "data"]
    )

    parse_tool = FunctionTool(
        parse_json, category=ToolCategory.UTILITY, tags=["parsing", "json"]
    )

    # Register tools
    print("\nRegistering tools:")
    registry.register(search_tool)
    registry.register(fetch_tool)
    registry.register(parse_tool)
    print("✓ Tools registered")

    # Get registry info
    print("\nRegistry Information:")
    info = registry.get_registry_info()
    print(f"  Total tools: {info['total_tools']}")
    print(f"  Tools by category:")
    for category, count in info["categories"].items():
        if count > 0:
            print(f"    • {category}: {count}")

    # Get tools by category
    print("\nTools by Category (DATA_RETRIEVAL):")
    data_tools = registry.get_tools_by_category(ToolCategory.DATA_RETRIEVAL)
    for tool in data_tools:
        schema = tool.get_schema()
        print(f"  • {schema.name}: {schema.description}")

    # Search tools
    print("\nSearching for tools with tag 'data':")
    found_tools = registry.search_tools("data")
    for tool in found_tools:
        schema = tool.get_schema()
        print(f"  • {schema.name} (tags: {', '.join(schema.tags)})")


def example_3_tool_validation():
    """Demonstrate tool parameter validation."""
    print("\n" + "=" * 60)
    print("Example 3: Tool Validation")
    print("=" * 60)

    def process_data(name: str, age: int, email: str) -> str:
        """Process user data."""
        return f"{name} ({age}) - {email}"

    tool = FunctionTool(process_data, category=ToolCategory.UTILITY)
    schema = tool.get_schema()

    print("\nTool Schema:")
    print(f"  Name: {schema.name}")
    print(f"  Parameters:")
    for param in schema.parameters:
        print(
            f"    • {param.name}: {param.type} {'(required)' if param.required else '(optional)'}"
        )

    # Valid parameters
    print("\nValidating Parameters:")
    print("-" * 60)

    print("\n1. Valid parameters:")
    try:
        valid_params = ToolValidator.validate_parameters(
            schema, name="Alice", age=30, email="alice@example.com"
        )
        print(f"✓ Valid: {valid_params}")
    except ValueError as e:
        print(f"✗ Error: {e}")

    # Invalid parameters (missing required)
    print("\n2. Missing required parameter:")
    try:
        invalid_params = ToolValidator.validate_parameters(
            schema,
            name="Bob",
            email="bob@example.com",
            # Missing 'age'
        )
        print(f"Result: {invalid_params}")
    except ValueError as e:
        print(f"✗ Error caught: {e}")

    # Type coercion
    print("\n3. Type coercion (string to int):")
    try:
        coerced_params = ToolValidator.validate_parameters(
            schema,
            name="Charlie",
            age="25",  # String, will be converted to int
            email="charlie@example.com",
        )
        print(
            f"✓ Coerced successfully: age={coerced_params['age']} (type: {type(coerced_params['age']).__name__})"
        )
    except Exception as e:
        print(f"✗ Error: {e}")


def example_4_tool_schemas():
    """Display and manage tool schemas."""
    print("\n" + "=" * 60)
    print("Example 4: Tool Schemas and Metadata")
    print("=" * 60)

    def extract_entities(text: str) -> List[str]:
        """Extract named entities from text."""
        return ["entity1", "entity2"]

    tool = FunctionTool(
        extract_entities,
        category=ToolCategory.UTILITY,
        version="2.1.0",
        author="NLP Team",
        tags=["nlp", "entities", "extraction"],
    )

    schema = tool.get_schema()

    print("\nTool Schema (JSON):")
    print("-" * 60)
    print(schema.to_json())


def example_5_dynamic_loading():
    """Demonstrate dynamic tool loading."""
    print("\n" + "=" * 60)
    print("Example 5: Dynamic Tool Loading")
    print("=" * 60)

    print("\nDynamic tool loading allows:")
    print("  • Loading tools from Python modules at runtime")
    print("  • Registering functions as tools dynamically")
    print("  • Building extensible agent systems")

    # Create registry and loader
    registry = ToolRegistry()
    loader = DynamicToolLoader(registry)

    print("\nExample usage:")
    print("""
# Load tools from a module
loader.load_from_module("my_tools_module")

# Load individual functions
def custom_tool(param: str) -> str:
    return f"Result: {param}"

loader.load_from_function(custom_tool)

# Verify loaded tools
print(f"Total tools: {len(registry.get_all_tools())}")
""")


def example_6_best_practices():
    """Show best practices for tools."""
    print("\n" + "=" * 60)
    print("Example 6: Best Practices for Tools")
    print("=" * 60)

    print("""
1. TOOL DESIGN:
   ✓ Single responsibility - one tool, one task
   ✓ Clear, descriptive names
   ✓ Comprehensive docstrings
   ✓ Type hints for all parameters
   ✗ Don't create tools that do multiple things
   ✗ Avoid ambiguous parameter names

2. PARAMETER VALIDATION:
   ✓ Validate all inputs before execution
   ✓ Provide clear error messages
   ✓ Use type hints for automatic validation
   ✓ Handle edge cases gracefully

3. DOCUMENTATION:
   ✓ Docstring explains what tool does
   ✓ List all parameters and return values
   ✓ Provide usage examples
   ✓ Include error conditions

4. CATEGORIZATION:
   ✓ Use meaningful categories for organization
   ✓ Tag tools for discovery
   ✓ Version your tools
   ✓ Track authorship

5. ERROR HANDLING:
   ✓ Catch and log errors
   ✓ Return meaningful error messages
   ✓ Don't let exceptions propagate unexpectedly
   ✓ Provide fallback values when appropriate

6. PERFORMANCE:
   ✓ Keep tools fast (<1 second ideal)
   ✓ Cache results when possible
   ✓ Use async for long operations
   ✓ Monitor tool usage metrics

Example of well-designed tool:

def calculate_discount(
    original_price: float,
    discount_percent: float
) -> float:
    \"\"\"
    Calculate discounted price.
    
    Args:
        original_price: Original price in dollars
        discount_percent: Discount percentage (0-100)
        
    Returns:
        Discounted price
        
    Raises:
        ValueError: If prices are negative
        
    Example:
        >>> calculate_discount(100.0, 20.0)
        80.0
    \"\"\"
    if original_price < 0 or discount_percent < 0:
        raise ValueError("Prices must be non-negative")
    
    discount_amount = original_price * (discount_percent / 100)
    return original_price - discount_amount
""")


if __name__ == "__main__":
    print("\n")
    print("█" * 60)
    print("█  LangChain Tools Integration - Examples")
    print("█" * 60)

    example_1_create_function_tool()
    example_2_tool_registry()
    example_3_tool_validation()
    example_4_tool_schemas()
    example_5_dynamic_loading()
    example_6_best_practices()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  • Tools wrap Python functions for agent use")
    print("  • Registries manage and organize tools")
    print("  • Validation ensures correct parameter usage")
    print("  • Schemas provide tool metadata to agents")
    print("  • Dynamic loading enables extensibility")
    print("=" * 60 + "\n")
