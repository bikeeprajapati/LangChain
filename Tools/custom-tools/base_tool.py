from langchain_core.tools import BaseTool
from typing import Type
from pydantic import Field

class MultiplyInput(BaseTool):
    a: float = Field(..., description="The first number to multiply.")
    b: float = Field(..., description="The second number to multiply.")


class MultiplyTool(BaseTool):
    name: str = "multiply_tool"
    description: str = "Multiply two numbers and return the result."
    args: Type[MultiplyInput] = MultiplyInput

    def _run(self, a: float, b: float) -> float:
        """Multiply two numbers and return the result."""
        return a * b
    
multiply_tool = MultiplyTool()
result = multiply_tool.invoke({"a": 3, "b": 4})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)
