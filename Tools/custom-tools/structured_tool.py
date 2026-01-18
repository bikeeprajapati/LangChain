from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: float = Field(..., description="The first number to multiply.")
    b: float = Field(..., description="The second number to multiply.")

def multiply_func(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b

multiply_tool = StructuredTool.from_function(
    func =multiply_func,
    name="multiply_tool",
    description="Multiply two numbers and return the result.",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({"a": 3, "b": 4})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)
