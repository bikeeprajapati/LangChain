from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation"
)

sample_text = """
LangChain is a framework for developing applications powered by language models.
It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.
And you can easily integrate it with other tools and data sources.
"""

chunks = text_splitter.split_text(sample_text)
print(chunks)

