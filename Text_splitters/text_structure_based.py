from langchain_text_splitters import  RecursiveCharacterTextSplitter

text = """LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.And youu can easily integrate it with other tools and data sources."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
)

chunks = splitter.split_text(text)
print(chunks)
print(len(chunks))