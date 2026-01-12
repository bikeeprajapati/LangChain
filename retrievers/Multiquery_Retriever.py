from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()

# -------------------------------
# Documents
# -------------------------------
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# -------------------------------
# Embeddings
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Vector Store
# -------------------------------
vectorstore = FAISS.from_documents(all_docs, embedding_model)

# -------------------------------
# Normal similarity retriever
# -------------------------------
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# -------------------------------
# LLM (HuggingFace)
# -------------------------------
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        temperature=0.3,
        max_new_tokens=256
    )
)

# -------------------------------
# MultiQuery Retriever
# -------------------------------
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# -------------------------------
# Query
# -------------------------------
query = "How to improve energy levels and maintain balance?"

similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

# -------------------------------
# Output
# -------------------------------
print("\n🔹 Similarity Retriever Results")
for i, doc in enumerate(similarity_results, 1):
    print(f"{i}. {doc.page_content}")

print("\n" + "*" * 80)

print("\n🔹 MultiQuery Retriever Results")
for i, doc in enumerate(multiquery_results, 1):
    print(f"{i}. {doc.page_content}")
