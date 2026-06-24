# AI Product Recommendation Chatbot

A conversational AI chatbot that recommends e-commerce products using natural 
language understanding and Retrieval-Augmented Generation (RAG). The system 
combines three memory strategies — sliding window, LLM-generated summarization, 
and vector-based semantic memory — to maintain coherent, context-aware 
conversations across multiple turns.

## Features
- 🔍 Semantic product search using sentence embeddings
- 🧠 Hybrid memory: sliding window + summary + vector retrieval (RAG)
- 🎯 Intent-based filtering for more relevant recommendations
- 💬 Multi-turn conversational context handling
- 🌐 Built with Python, leveraging an LLM API for response generation

## Tech Stack
- Python
- Sentence-Transformers (embeddings)
- FAISS (vector similarity search)
- Large Language Model API (e.g., Google Gemini)
