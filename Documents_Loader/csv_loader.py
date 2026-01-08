from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="/home/bikee-prajapati/LangChain_Models/Documents_Loader/Social_Network_Ads.csv")
docs = loader.load()
print(docs[0].page_content)