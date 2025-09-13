# Advanced RAG Chatbot Application Documentation

## Table of Contents
1. [Overview](#overview)
2. [Enhanced Features](#enhanced-features)
3. [Architecture](#architecture)
4. [Data Flow](#data-flow)
5. [Component Integration](#component-integration)
6. [Installation and Setup](#installation-and-setup)
7. [Running the Application](#running-the-application)
8. [Module Execution Procedures](#module-execution-procedures)
9. [Environment Configuration](#environment-configuration)
10. [Dependency Management](#dependency-management)
11. [Troubleshooting](#troubleshooting)
12. [Additional Information](#additional-information)

## Overview

This Advanced RAG (Retrieval-Augmented Generation) Chatbot is an intelligent research assistant built with Python, Streamlit, and LangChain. It allows users to upload PDF research papers, search and download papers from Arxiv, and ask sophisticated questions about the content. The application processes complex PDF elements including mathematical equations, tables, figures, and bibliographic references.

## Enhanced Features

1. **Advanced PDF Processing**: Extracts mathematical equations, tables, figures, and bibliographic references
2. **Arxiv Integration**: Search and download research papers directly from Arxiv
3. **Specialized Query Processing**: Handles content retrieval, summarization, and quantitative result extraction
4. **Enhanced UI**: Improved interface with Arxiv search functionality

## Architecture

The application follows a modular architecture with the following components:

1. **Frontend**: Streamlit-based web interface (`app.py`)
2. **Backend Services**:
   - Advanced Data Ingestion (`utils/advanced_data_ingest.py`)
   - Vector Store Management (`utils/vector_store.py`)
   - Advanced RAG Chain Processing (`utils/advanced_rag_chain.py`)
3. **Data Storage**: ChromaDB vector database
4. **External Services**: OpenRouter API for LLM inference, Arxiv API for paper search

## Data Flow

The application follows a comprehensive data flow that processes documents, stores them in a vector database, and retrieves relevant information to answer user queries. Here's a detailed breakdown:

### 1. Document Upload and Processing
```
User uploads PDF/DOCX → Streamlit frontend → advanced_data_ingest.py →
Multi-method Extraction (pdfplumber, pdfminer) →
Specialized Content Extraction (equations, tables, figures, bibliography) →
Document Chunks with Enhanced Metadata → Vector Store
```

### 2. Arxiv Paper Search and Download
```
User search query → Streamlit frontend → advanced_data_ingest.py →
Arxiv API Search → Paper Results → User Selection →
Paper Download → Processing Pipeline → Vector Store
```

### 3. Vector Database Creation and Management
```
Enhanced Document Chunks → vector_store.py →
HuggingFace Embeddings (all-MiniLM-L6-v2) →
ChromaDB Vector Storage → Persistent Storage
```

### 4. Query Processing and Response Generation
```
User Query → Streamlit frontend → advanced_rag_chain.py →
Query Type Detection → Specialized Prompt Selection →
ChromaDB Retriever (top 5 matches) → Relevant Documents →
OpenRouter LLM (GPT-3.5-turbo) → Context-Aware Response
```

### 5. Conversation Management and Context Learning
```
User Query/Response → Session State →
Vector Store (for context learning) →
UI Display with Source Documents
```

## Component Integration

### Frontend-Backend Integration Points

1. **Document Processing**:
   - Frontend (`app.py`) calls `load_and_process_advanced()` from `utils/advanced_data_ingest.py`
   - Frontend calls `build_vector_db()` or `load_vector_db()` from `utils/vector_store.py`
   - Frontend calls `create_advanced_rag_chain()` from `utils/advanced_rag_chain.py`

2. **Arxiv Integration**:
   - Frontend calls `search_arxiv_papers()` from `utils/advanced_data_ingest.py`
   - Frontend calls `download_arxiv_paper()` from `utils/advanced_data_ingest.py`

3. **Query Processing**:
   - Frontend calls `process_specialized_query()` from `utils/advanced_rag_chain.py`
   - Results are displayed in the chat interface

4. **State Management**:
   - Streamlit's `st.session_state` manages conversations, vector store, and RAG chain
   - Vector database persistence through ChromaDB

### Module Interactions

1. **app.py** ↔ **utils/advanced_data_ingest.py**:
   - `app.py` calls `load_and_process_advanced()` to process uploaded PDFs
   - `app.py` calls `search_arxiv_papers()` and `download_arxiv_paper()` for Arxiv integration

2. **app.py** ↔ **utils/vector_store.py**:
   - `app.py` calls `build_vector_db()`, `load_vector_db()`, `save_vector_db()`, `add_to_vector_db()`
   - Vector store object is stored in session state for reuse

3. **app.py** ↔ **utils/advanced_rag_chain.py**:
   - `app.py` calls `create_advanced_rag_chain()` to initialize the RAG pipeline
   - `app.py` calls `process_specialized_query()` for query processing
   - RAG chain object is stored in session state for query processing

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone or download the repository
2. Navigate to the project directory:
   ```bash
   cd RAG-ASSIGNMENT-main/RAG-PROJECT-main/RAG-PROJECT-main/rag_streamlit_chatbot
   ```

3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies Explained

- `streamlit`: Web framework for the frontend interface
- `langchain`: Core framework for building LLM applications
- `langchain-community`: Community-contributed components for LangChain
- `langchain-huggingface`: HuggingFace integration for embeddings
- `python-dotenv`: Environment variable management
- `pypdf`: Basic PDF document processing
- `pdfplumber`: Advanced PDF processing (tables, images, layout)
- `pdfminer.six`: PDF text extraction
- `arxiv`: Arxiv API client
- `chromadb`: Vector database for storing document embeddings
- `requests`: HTTP client for API calls
- `numpy`: Numerical computing
- `pandas`: Data manipulation

## Running the Application

### Starting the Application

1. Ensure you have set up your environment variables (see Environment Configuration below)
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. The application will open in your default web browser at `http://localhost:8501`

### Application Workflow

1. **Initial Load**:
   - Application attempts to load any existing vector database
   - If found, displays loaded documents in the sidebar

2. **Document Upload**:
   - Use the file uploader in the sidebar to add PDF research papers
   - Documents are processed with enhanced extraction and added to the vector database
   - Processed documents are displayed in the sidebar

3. **Arxiv Search**:
   - Enter keywords in the Arxiv search box
   - View search results with paper titles, authors, abstracts
   - Download papers directly to the application

4. **Chat Interaction**:
   - Once documents are processed, the chat interface becomes active
   - Ask questions about your documents in the chat input
   - Responses will be displayed with source document information
   - Specialized queries are automatically detected and processed:
     - Content retrieval ("What does Paper X conclude?")
     - Summarization ("Summarize the methodology of Paper C")
     - Quantitative extraction ("Report the accuracy and F1 scores from Paper D")

## Module Execution Procedures

### app.py (Main Application)

This is the Streamlit frontend that orchestrates all components and provides the user interface:

1. **Initialization**:
   - Loads environment variables from `.env` file
   - Sets up Streamlit page configuration with custom styling
   - Initializes session state variables for conversations, vector store, and RAG chain
   - Automatically loads existing vector database on startup

2. **Document Processing**:
   - Handles file uploads for PDF and DOCX documents
   - Calls `load_and_process_advanced()` from `advanced_data_ingest.py` for document processing
   - Calls `build_vector_db()` or `load_vector_db()` from `vector_store.py` to manage vector storage
   - Calls `create_advanced_rag_chain()` from `advanced_rag_chain.py` to initialize the RAG pipeline
   - Manages progress indicators and error handling during processing

3. **Arxiv Integration**:
   - Provides interface for searching Arxiv papers
   - Calls `search_arxiv_papers()` from `advanced_data_ingest.py` to search for papers
   - Handles paper downloads with `download_arxiv_paper()` and adds them to the document processing pipeline
   - Displays search results with paper metadata in expandable sections

4. **Query Processing**:
   - Receives user input from chat interface
   - Calls `process_specialized_query()` from `advanced_rag_chain.py` for query processing
   - Displays results with source documents and metadata
   - Manages conversation history and context learning
   - Handles conversation naming and management

5. **UI Components**:
   - Sidebar for document management, Arxiv search, and conversation history
   - Main chat interface for user interactions
   - Custom CSS styling for enhanced user experience
   - Progress indicators and error messaging

**Key Features**:
- Session state management for persistent conversations
- Automatic vector database loading and saving
- Real-time progress feedback during document processing
- Responsive UI with custom styling
- Error handling and user-friendly messaging

### utils/advanced_data_ingest.py

Handles advanced PDF and DOCX document loading with enhanced extraction capabilities and Arxiv integration:

1. **extract_with_pdfplumber(pdf_path)**:
   - Uses `pdfplumber` to extract text, tables, and image locations
   - Returns content, tables, and images data
   - Preserves document layout and structure information

2. **extract_mathematical_equations(text)**:
   - Extracts mathematical equations using regex patterns
   - Supports LaTeX and other equation formats
   - Handles both inline and display math modes

3. **extract_bibliography(text)**:
   - Extracts bibliography/references section from text
   - Uses pattern matching for common bibliography formats
   - Preserves citation information for context

4. **extract_figures_and_captions(text)**:
   - Extracts figure references and captions
   - Uses pattern matching for figure references
   - Maintains figure context for better understanding

5. **extract_sections(text)**:
   - Extracts section headings and structure
   - Supports multi-level section numbering
   - Helps organize document content hierarchically

6. **extract_text_from_docx(docx_path)**:
   - Extracts text content from DOCX files
   - Handles paragraphs and table content
   - Converts document structure to plain text

7. **load_and_process_advanced(file_path)**:
   - Main document processing function supporting both PDF and DOCX
   - Combines all extraction methods based on file type
   - Creates enhanced document with metadata
   - Uses `RecursiveCharacterTextSplitter` for chunking
   - Returns list of document chunks with enhanced metadata

8. **search_arxiv_papers(query, max_results)**:
   - Searches Arxiv for papers matching query
   - Returns paper metadata (title, authors, abstract, etc.)
   - Supports relevance-based sorting

9. **download_arxiv_paper(paper_entry_id, output_dir)**:
   - Downloads a paper from Arxiv by entry ID
   - Saves PDF to specified directory
   - Integrates with document processing pipeline

**Key Features**:
- Multi-format document support (PDF and DOCX)
- Enhanced content extraction (equations, tables, figures, bibliography)
- Metadata preservation for better context
- Arxiv integration for paper discovery and download
- Document chunking with overlap for better retrieval

**Dependencies**:
- `pdfplumber`: Advanced PDF processing
- `pdfminer.high_level`: Text extraction
- `arxiv`: Arxiv API client
- `langchain.text_splitter.RecursiveCharacterTextSplitter`
- `python-docx`: DOCX document processing

### utils/vector_store.py

Manages ChromaDB vector database operations for document storage and retrieval:

1. **build_vector_db(documents, persist_directory)**:
   - Creates embeddings using `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2` model
   - Builds ChromaDB vector store from documents
   - Creates persistent storage for document embeddings
   - Returns vector store object for further operations

2. **load_vector_db(persist_directory)**:
   - Loads existing ChromaDB vector store from persistent storage
   - Initializes embedding model for similarity searches
   - Returns vector store object for querying

3. **save_vector_db(vectorstore)**:
   - Persists vector store to disk
   - Ensures data durability between application sessions
   - Updates storage with new or modified embeddings

4. **add_to_vector_db(vectorstore, documents)**:
   - Adds new documents to existing vector store
   - Updates persistent storage with new embeddings
   - Maintains consistency of vector database

5. **get_source_documents(vectorstore)**:
   - Retrieves list of source document names from vector store
   - Extracts unique document sources for UI display
   - Helps users understand which documents are available for querying

**Key Features**:
- Persistent storage using ChromaDB
- HuggingFace embeddings with `all-MiniLM-L6-v2` model
- Session persistence for document collections
- Source document tracking for transparency
- Integration with LangChain vector store interface

**Dependencies**:
- `langchain_community.vectorstores.Chroma`: Vector database implementation
- `langchain_huggingface.HuggingFaceEmbeddings`: Embedding model integration
- `chromadb.config.Settings`: Database configuration

### utils/advanced_rag_chain.py

Implements the advanced Retrieval-Augmented Generation pipeline with specialized query processing:

1. **OpenRouterLLM Class**:
   - Custom LLM wrapper for OpenRouter API
   - Implements LangChain's `LLM` base class
   - Handles API calls to OpenRouter with proper headers
   - Supports configurable models, temperature, and max tokens

2. **get_llm()**:
   - Factory function to get OpenRouter LLM instance
   - Handles API key validation from environment variables
   - Configures LLM with default parameters
   - Supports optional site URL and name for API tracking

3. **create_advanced_rag_chain(vectorstore)**:
   - Initializes OpenRouter LLM with API key
   - Creates retriever from vector store with top-5 similarity search
   - Sets up specialized prompt templates for different query types
   - Creates `RetrievalQA` chain with source document return
   - Returns chain object for query processing

4. **process_specialized_query(rag_chain, query)**:
   - Detects query type (general, summary, extraction) using keyword analysis
   - Applies appropriate prompt template based on query type
   - Processes query with RAG chain
   - Returns result with source documents

**Specialized Query Processing**:
- **Summarization queries**: Detected by keywords like "summarize" or "summary"
- **Quantitative extraction**: Detected by keywords like "accuracy", "f1", or "score"
- **Content extraction**: Detected by keywords like "conclude" or "methodology"
- **General queries**: Default processing for all other queries

**Prompt Templates**:
- **Base template**: General question answering with context
- **Summary template**: Focused on concise content summarization
- **Extraction template**: Precise information extraction from context

**Key Features**:
- Integration with OpenRouter API for LLM inference
- Specialized prompt templates for different query types
- Automatic query type detection
- Source document attribution in responses
- Configurable LLM parameters

**Dependencies**:
- `langchain.chains.RetrievalQA`: RAG chain implementation
- `langchain.prompts.PromptTemplate`: Prompt template management
- `langchain.llms.base.LLM`: LLM base class
- `requests`: For API calls to OpenRouter

## Environment Configuration

### .env File Setup

Create a `.env` file in the project root with your OpenRouter API key:

```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional site information for OpenRouter
OPENROUTER_SITE_URL=https://your-site-url.com
OPENROUTER_SITE_NAME=QA-AI-Agent
```

### API Key Requirements

1. **OpenRouter API Key**:
   - Required for LLM inference
   - Sign up at [OpenRouter](https://openrouter.ai/) to get an API key
   - The application uses the `openai/gpt-3.5-turbo` model by default

### Environment Variables Used

- `OPENROUTER_API_KEY`: API key for OpenRouter service
- `OPENROUTER_SITE_URL` (optional): Site URL for API tracking
- `OPENROUTER_SITE_NAME` (optional): Site name for API tracking

## Dependency Management

### Requirements File

The `requirements.txt` file contains all necessary dependencies:

```
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-huggingface>=0.0.1
python-dotenv>=1.0.0
pypdf>=3.17.0
pdfplumber>=0.10.0
pdfminer.six>=20221105
arxiv>=2.1.0
chromadb>=0.4.22
requests>=2.31.0
numpy>=1.24.0
pandas>=2.0.0
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Updating Dependencies

To update dependencies, modify the `requirements.txt` file and run:

```bash
pip install -r requirements.txt --upgrade
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "OPENROUTER_API_KEY not found" Error
**Problem**: API key is missing or not properly configured
**Solution**: 
- Ensure you have created a `.env` file with your API key
- Verify the API key is correct and active
- Restart the application after adding the API key

#### 2. "Error processing documents" Error
**Problem**: Issues with PDF processing or vector database creation
**Solutions**:
- Check that uploaded files are valid PDFs
- Ensure sufficient disk space for vector database storage
- Verify permissions for the `vector_dbs` directory
- Try uploading a different PDF file

#### 3. "Error processing your question" Error
**Problem**: Issues with the RAG chain or API call
**Solutions**:
- Check your internet connection
- Verify your OpenRouter API key is valid and has credits
- Try rephrasing your question
- Check the application logs for more detailed error information

#### 4. Arxiv Search Issues
**Problem**: Problems searching or downloading papers from Arxiv
**Solutions**:
- Check your internet connection
- Try a different search query
- Verify the paper entry ID is correct
- Check if Arxiv is experiencing service issues

#### 5. Application Not Starting
**Problem**: Streamlit server fails to start
**Solutions**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that port 8501 is not being used by another application
- Try running with a different port: `streamlit run app.py --server.port 8502`

#### 6. Vector Database Issues
**Problem**: Problems loading or creating vector database
**Solutions**:
- Delete the `vector_dbs` directory and restart the application
- Ensure the application has write permissions to the directory
- Check available disk space

### Debugging Tips

1. **Enable Debug Logging**:
   - The application includes debug logging in the modules
   - Check the terminal output for detailed information

2. **Check File Permissions**:
   - Ensure the application can read/write to the `vector_dbs` and `downloads` directories

3. **Verify Dependencies**:
   - Run `pip list` to check installed packages
   - Ensure all required packages are installed with correct versions

4. **Test API Key**:
   - Verify your OpenRouter API key is working by testing it directly with curl:
   ```bash
   curl -X POST https://openrouter.ai/api/v1/chat/completions \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "openai/gpt-3.5-turbo",
       "messages": [{"role": "user", "content": "Hello"}]
     }'
   ```

### Performance Considerations

1. **Large Documents**:
   - Processing large PDFs may take time
   - Advanced extraction features may increase processing time
   - Consider splitting very large documents into smaller sections

2. **Memory Usage**:
   - The application loads all document embeddings into memory
   - For very large document collections, consider using a dedicated vector database server

3. **API Costs**:
   - Each query consumes OpenRouter API credits
   - Monitor your API usage to avoid unexpected costs

4. **Arxiv Rate Limits**:
   - Arxiv has rate limits on API requests
   - Avoid making too many requests in a short period

## Additional Information

### Data Persistence

- Vector databases are stored in the `vector_dbs` directory
- Downloaded Arxiv papers are stored in the `downloads` directory
- Uploaded document content is not stored directly (only embeddings)
- Conversation history is stored in Streamlit session state (temporary)

### Security Considerations

- API keys are loaded from environment variables, not hardcoded
- No user data is sent to external services except for LLM inference
- Document content is processed locally
- Downloaded papers are stored locally

### Customization

- Modify the prompt templates in `utils/advanced_rag_chain.py` to change the assistant's behavior
- Adjust chunk size and overlap in `utils/advanced_data_ingest.py` for different document types
- Change the embedding model in `utils/vector_store.py` for different performance characteristics
- Add new query type detection in `process_specialized_query()` function

### utils/data_ingest.py

Provides basic PDF document loading functionality using PyPDFLoader:

1. **load_and_split(pdf_path)**:
   - Loads PDF documents using PyPDFLoader
   - Splits documents into chunks using RecursiveCharacterTextSplitter
   - Returns list of document chunks for vector storage

**Key Features**:
- Simple PDF processing pipeline
- Document chunking with overlap for better context
- Error handling for missing or invalid files

**Dependencies**:
- `langchain_community.document_loaders.PyPDFLoader`: PDF document loading
- `langchain.text_splitter.RecursiveCharacterTextSplitter`: Document chunking

### utils/rag_chain.py

Implements a basic Retrieval-Augmented Generation pipeline:

1. **OpenRouterLLM Class**:
   - Custom LLM wrapper for OpenRouter API
   - Implements LangChain's `LLM` base class
   - Handles API calls to OpenRouter

2. **create_rag_chain(vectorstore)**:
   - Initializes OpenRouter LLM with API key
   - Creates retriever from vector store with top-3 similarity search
   - Sets up prompt template for general question answering
   - Creates `RetrievalQA` chain
   - Returns chain object

**Key Features**:
- Integration with OpenRouter API for LLM inference
- Basic prompt template for question answering
- Source document retrieval in responses
- Environment variable configuration

**Dependencies**:
- `langchain.chains.RetrievalQA`: RAG chain implementation
- `langchain.prompts.PromptTemplate`: Prompt template management
- `langchain.llms.base.LLM`: LLM base class
- `requests`: For API calls to OpenRouter