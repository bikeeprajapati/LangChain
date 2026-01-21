import os
import requests
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

# ENV 
load_dotenv()

# TOOLS 

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a given city.
    """
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return "Weather API key is missing."

    url = "http://api.weatherstack.com/current"
    params = {
        "access_key": api_key,
        "query": city,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "current" not in data:
            return f"Could not retrieve weather for {city}"

        current = data["current"]
        return (
            f"Weather in {city}: "
            f"{current['temperature']}°C, "
            f"{current['weather_descriptions'][0]}, "
            f"Humidity {current['humidity']}%"
        )

    except Exception as e:
        return f"Weather API error: {e}"

# LLM 

# First create the base endpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7,
)

# Wrap it in ChatHuggingFace to enable tool binding
chat_model = ChatHuggingFace(llm=llm)

# AGENT 
tools = [search_tool, get_weather_data]

# Use the new create_agent from langchain.agents
agent_executor = create_agent(
    model=chat_model,
    tools=tools,
)

# RUN

query = (
    "Find the capital of Nepal, "
    "how much is the probability for raining in next 2 hours."
)

response = agent_executor.invoke({"messages": [("user", query)]})
print(response)