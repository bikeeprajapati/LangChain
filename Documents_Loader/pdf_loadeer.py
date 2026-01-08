from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("/home/bikee-prajapati/LangChain_Models/Documents_Loader/dl-curriculum.pdf")
docs = loader.load()
print(docs[0].page_content)
