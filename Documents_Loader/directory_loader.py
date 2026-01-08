from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

loader  = DirectoryLoader(
    path="/home/bikee-prajapati/LangChain_Models/Documents_Loader/Books",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(len(docs))
print(docs[0].page_content) 
print(docs[0].metadata)