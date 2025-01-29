# JuscticeAI: LLM-based Legal ChatBot

![Python 3.12](https://img.shields.io/badge/Python-3.10-brightgreen.svg) [![ChatGPT](https://img.shields.io/badge/ChatGPT-74aa9c?logo=openai&logoColor=white)](#)  

JusticeAI is an advanced chatbot built on Large Language Models (LLM), designed to deliver legal insights. Utilizing the Retrieval-Augmented Generation (RAG) framework, along with cutting-edge language models and embeddings, the bot retrieves and generates relevant answers from a curated legal document repository. This project is focused on Nigerian law and other related legal content.

## Table of Contents

- [Overview](#introduction)
- [Key Features](#features)
- [System Architecture](#architecture)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#usage)
- [Live Demo](#deployed-website)


## Introduction

JuscticeAI aims to assist users by providing accurate and concise legal information in the country on nigeria  and related legal documents. The chatbot retrieves relevant context from the knowledge base to answer user queries efficiently.

## Features

- Interactive chatbot for obtaining legal information
- Utilizes FAISS for fast and efficient vector searches
- Embeds documents using Google’s Generative AI Embeddings
- Efficient handling of large document sets with splitting and batching
- Provides source citations for the retrieved information


## Architecture

The architecture of JuscticeAI includes the following components:

1. **Document Loader**: Loads legal documents from a specified directory containing PDF files.
2. **Text Splitter**: Breaks down large documents into smaller chunks for efficient embedding.
3. **Embeddings**: Transforms text into vector representations using Google Generative AI Embeddings.
4. **Vector Store**: FAISS is used for storing and retrieving document embeddings.
5. **LLM**: The ChatGroq API generates responses based on retrieved documents and user queries.
6. **Memory**: Keeps track of the conversation history for enhanced context in ongoing chats.

## Setup and Installation

### Prerequisites

- Python 3.12
- [Streamlit](https://streamlit.io/)
- [LangChain Community](https://github.com/langchain-ai/langchain-community)
- [Google Generative AI](https://github.com/google-research/google-research/tree/master/large-scale-causal-ml)
- [FAISS](https://github.com/facebookresearch/faiss)

### Installation Steps

1. **Clone the Repository**

```bash
   git clone https://github.com/ikehi/ClarityBot
   cd JuscticeAI
```

2.  **Set Up and Activate Virtual Environment**

```bash
    conda create -p venv python==3.12
    conda activate C:\directory\venv
```

3. **Install Dependencies**

```bash
    pip install -r requirements.txt
```

4. **Set Up Environment Variables**

Create a .env file in the project root directory and add your API keys:
```bash
    GOOGLE_API_KEY=your_google_api_key
    GROQ_API_KEY=your_groq_api_key
```

5. **Split, Embed and Save Documents**

Run the following script to load, split, embed, and save your legal documents:
```bash
    python ingestion.py
```

## Usage
Run the Streamlit Application

```bash
streamlit run app.py
```
## Deployed Website

JuscticeAI is also deployed on Streamlit Cloud. You can access the chatbot directly via the following link:https://justiceal.streamlit.app/





