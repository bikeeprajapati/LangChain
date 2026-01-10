from langchain_text_splitters import RecursiveCharacterTextSplitter , Language
text = """
            def add(a, b):
                return a + b
            class Calculator:
                def multiply(self, a, b):
                    return a * b
            """
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=10,
    chunk_overlap=0,
)
chunks = splitter.split_text(text)
print(chunks)