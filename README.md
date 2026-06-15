# 🎓 CampusGPT: AI-Powered College Assistant

CampusGPT is an intelligent **Retrieval-Augmented Generation (RAG)** based college assistant that enables students to interact with their study materials using natural language. Students can upload multiple PDF documents and ask questions related to their notes, receiving accurate answers along with source citations.

---

## 🚀 Features

* 📄 **Multi-PDF Upload Support**
* 🔍 **Semantic Search using Vector Embeddings**
* 🤖 **AI-Powered Question Answering using Llama 3.3 (Groq API)**
* 🧠 **Conversation Memory for Follow-up Questions**
* 📚 **Source Citations with Chunk References**
* 💬 **ChatGPT-like Interactive Interface**
* 📊 **PDF Statistics and Chunk Preview**
* 🗑️ **Clear Chat Functionality**
* ♻️ **Reset Knowledge Base Option**
* ⚡ **Fast Inference with Groq**

---

## 🏗️ System Architecture

```text
PDF Upload
    ↓
Text Extraction (PyPDF)
    ↓
Text Chunking
    ↓
Embeddings Generation
    ↓
ChromaDB Vector Storage
    ↓
Semantic Retrieval
    ↓
Conversation Memory
    ↓
Llama 3.3 via Groq API
    ↓
Answer Generation with Source Attribution
```

---

## 🛠️ Tech Stack

| Component             | Technology                                 |
| --------------------- | ------------------------------------------ |
| Frontend              | Streamlit                                  |
| Language              | Python                                     |
| LLM Provider          | Groq                                       |
| Model                 | Llama 3.3 70B Versatile                    |
| Embeddings            | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database       | ChromaDB                                   |
| PDF Processing        | PyPDF                                      |
| Text Chunking         | LangChain Text Splitters                   |
| Environment Variables | python-dotenv                              |
| Version Control       | Git & GitHub                               |

---

## 📂 Project Structure

```text
CampusGPT/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
├── test_groq.py
├── chroma_db/
└── assets/
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CampusGPT.git
cd CampusGPT
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📸 Screenshots

Add screenshots of:

* PDF Upload Interface
* Chat Interface
* Source Citations
* Conversation Memory

---

## 🎯 Key Features Demonstrated

* Retrieval-Augmented Generation (RAG)
* Vector Similarity Search
* Conversational AI
* Source Attribution
* Multi-document Knowledge Base
* Context-aware Question Answering

---

## 📈 Future Enhancements

* User Authentication
* Voice Input Support
* Cloud Deployment
* Attendance Prediction Module
* Exam Question Generator
* Placement Preparation Assistant
* Course Recommendation System

---

## 🎓 Academic Use

This project was developed as a **Final Year Project** to demonstrate the practical implementation of modern **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** systems in educational applications.

---

## 👨‍💻 Author

**Aditya Kumar Sharma**

* GitHub: https://github.com/Aditya529-ux
* LinkedIn: Add your LinkedIn profile here

---

## 📜 License

This project is licensed under the **MIT License**.

Feel free to use, modify, and distribute this project for educational purposes.

---

⭐ If you found this project useful, consider giving it a star on GitHub!
