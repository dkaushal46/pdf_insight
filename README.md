# 📄 PDFInsight – Chat with PDF using LLM

## 🚀 Project Overview
**PDFInsight** is a simple and powerful tool that enables users to interact with PDF documents using natural language queries. Powered by **LangChain**, **OpenAI embeddings**, and **FAISS**, this app allows semantic search over PDF content and provides accurate answers to user questions.

Built with an intuitive **Streamlit** interface, users can upload PDF files, generate embeddings, and chat with their documents in real-time.

---

## 🔥 Features
- ✅ Upload PDF files and parse their content.
- ✅ Generate embeddings from PDF text using **OpenAI embeddings**.
- ✅ Stores embeddings locally for faster subsequent searches.
- ✅ Uses **FAISS** for efficient vector-based similarity search.
- ✅ Query the document using natural language.
- ✅ Simple web-based interface powered by **Streamlit**.
- 💾 Embedding persistence: embeddings are saved as `.pkl` files locally.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** Streamlit
- **Libraries:**
  - LangChain
  - OpenAI API
  - FAISS
  - PyPDF2
  - dotenv
  - Streamlit Extras

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YourUsername/PDFInsight.git
cd PDFInsight
