from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "LangChain enables developers to build applications that can understand and generate human-like text.",
    "LangChain provides tools for working with language models, including prompt management and memory.",
]

document_embeddings = embeddings.embed_documents(documents)    

query = "What is LangChain?"
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)[0]


index, score = max(enumerate(scores), key=lambda x: x[1])


print("Query:", query)
print("Most similar document:", documents[index])
print("Similarity score:", score)
