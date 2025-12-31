from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "The sentiment of the review (positive, negative, neutral)"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The movie was fantastic! The plot was engaging and the characters were well-developed. I especially loved the cinematography.""")
print(result)
