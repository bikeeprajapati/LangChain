from langchain_core.tools import tool

#Step 1: Create a Function
def multiiply(a,b):
    """Multiply two numbers and return the result."""
    return a * b

#Step 2: Add type hints
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b

#Step 3: add tool decorator
@tool
def multiply_tool(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b

result = multiply_tool.invoke({"a": 3, "b": 4})
print(result)

print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)