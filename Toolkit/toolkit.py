from langchain_core.tools import tool

#Custom tool
@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers and return the result."""
    return a - b
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b
@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers and return the result."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b
class math_tools:
    def get_tools(self):
        return [add, subtract, multiply, divide]
