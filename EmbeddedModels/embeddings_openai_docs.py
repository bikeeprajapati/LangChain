from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It enables developers to build applications that can interact with various data sources.",
    "LangChain supports integration with multiple language model providers."
]

result = embeddings.embed_documents(documents)
print(str(result))