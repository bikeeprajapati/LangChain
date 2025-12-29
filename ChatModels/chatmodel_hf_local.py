from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float32  
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=False,
    repetition_penalty=1.03
)

llm = HuggingFacePipeline(pipeline=pipe)
chat = ChatHuggingFace(llm=llm)

print("Generating response...\n")
result = chat.invoke("What is the capital of Nepal?")
print(result.content)
