from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain this in simple terms, what is {topic}?")
])
prompt = chat_template.invoke({'domain':"research", 'topic':"Attention Is All You Need"})
print(prompt)