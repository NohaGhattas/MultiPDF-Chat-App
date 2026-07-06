# MultiPDF Chat App 📚🤖

## Introduction
------------
The MultiPDF Chat App is a powerful Python application that allows you to have an interactive conversation with multiple PDF documents simultaneously. You can ask questions about your documents using natural language, and the application will instantly provide relevant, context-aware responses. 

By leveraging the power of **Google Gemini (1.5 Flash)** and **Google GenAI Embeddings**, the app ensures highly accurate, fast, and completely cost-effective local execution without complex local package constraints.

## How It Works
------------

The application follows a structured RAG (Retrieval-Augmented Generation) workflow to provide precise answers:

1. **PDF Text Extraction:** The app reads multiple uploaded PDF documents and extracts their raw text content sequentially.
2. **Text Chunking:** The extracted text is split into smaller, manageable chunks using a semantic character splitter to retain context.
3. **Vector Embeddings:** The application utilizes Google GenAI's `embedding-001` model to generate semantic vector representations of these text chunks.
4. **Vector Storage:** The generated embeddings are indexed locally into a **FAISS** (Facebook AI Similarity Search) vector database for ultra-fast retrieval.
5. **Contextual Querying:** When you ask a question, the app searches the FAISS index to extract the most semantically relevant text chunks.
6. **Response Generation:** The selected chunks, along with your chat history, are passed to **Gemini-1.5-Flash** to formulate a natural, accurate response.

# MultiPDF Chat App 📚🤖

<img src="https://i.postimg.cc/FfGXJcHy/localhost.jpg" alt="MultiPDF Chat App Interface" width="90%">

## Introduction
------------
The MultiPDF Chat App is a powerful Python application that allows you to have an interactive conversation with multiple PDF documents simultaneously. You can ask questions about your documents using natural language, and the application will instantly provide relevant, context-aware responses. 

By leveraging the power of **Google Gemini (1.5 Flash)** and **Google GenAI Embeddings**, the app ensures highly accurate, fast, and completely cost-effective local execution without complex local package constraints.

## How It Works
------------

<img src="https://i.postimg.cc/NybvrTjV/PDF-Lang-Chain.jpg" alt="MultiPDF Chat App Diagram" width="85%">

The application follows a structured RAG (Retrieval-Augmented Generation) workflow to provide precise answers...

## Dependencies and Installation
----------------------------
To set up and run the MultiPDF Chat App locally, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/NohaGhattas/ask-multiple-pdfs-main.git](https://github.com/NohaGhattas)
cd ask-multiple-pdfs-main
