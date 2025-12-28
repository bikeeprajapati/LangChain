
# LangChain Study 📘

This repository contains my **hands-on learning and practice code for LangChain**, focusing on building LLM-powered applications using **Google Gemini**, prompt engineering, and basic chains.

The goal of this project is to understand **how LangChain works internally** and how to integrate it with modern Large Language Models (LLMs).

---

## 🚀 Technologies Used

- **Python 3.10+**
- **LangChain**
- **Google Gemini (Gemini 1.5 Turbo)**
- **dotenv** for environment variables
- **Git & GitHub**

---

## 📂 Project Structure

```

langchain-study/
│
├── .gitignore
├── .env                # API keys (ignored)
├── main.py             # Basic LangChain + Gemini example
├── requirements.txt
└── README.md

````

---

## 🔑 Environment Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/langchain-study.git
cd langchain-study
````

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 API Key Setup

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

⚠️ **Never push `.env` to GitHub**

---

## 🧪 Example Code (Gemini + LangChain)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-turbo",
    temperature=0.3
)

response = model.invoke("What is the capital of France?")
print(response.content)
```

### ✅ Output

```
Paris
```

---

## 📌 Topics Covered (Ongoing)

* ✔️ LangChain basics
* ✔️ Chat models
* ✔️ Prompt invocation
* ⏳ Prompt templates
* ⏳ Chains
* ⏳ Memory
* ⏳ Retrieval-Augmented Generation (RAG)

---

## 🎯 Learning Goals

* Understand LangChain architecture
* Learn how to interact with LLMs programmatically
* Build AI-powered applications step by step
* Prepare foundation for **AI Engineer / Backend AI roles**

---

## 🤝 Contributions

This is a personal learning repository, but suggestions and improvements are welcome.

---

## 📜 License

This project is for **educational purposes**.

---

## ✨ Author

**Vicky Prajapati**
Computer Science Student | AI & Backend Enthusiast
🇳🇵 Nepal

```

---

If you want, I can also:
- ✅ Make it **more beginner-friendly**
- ✅ Convert it into **portfolio-ready README**
- ✅ Add **badges (Python, LangChain, Gemini)**
- ✅ Add **future roadmap section**

Just tell me 👍
```
