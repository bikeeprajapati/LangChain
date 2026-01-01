from pydantic import BaseModel, Field, EmailStr,Field
from typing import Optional


class User(BaseModel):
    id: int = Field(..., gt=0, description="The unique identifier for the user")
    name: str = 'Vicky'
    age: Optional[int] = Field(..., ge=0, description="The age of the user")
    email: EmailStr = Field(..., description="The email address of the user")
    SGPA: Optional[float] = Field(None, ge=0.0, le=10.0, description="The SGPA of the user")

new_user_data = {
    "id": 1,
    "name": "",
    "age": None,
    "email":"abc@gmail.com",
    "SGPA": None
}

user = User(**new_user_data)
print(user)
