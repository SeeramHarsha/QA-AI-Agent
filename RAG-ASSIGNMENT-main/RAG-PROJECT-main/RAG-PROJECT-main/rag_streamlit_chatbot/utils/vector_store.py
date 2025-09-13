from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
import os


def build_vector_db(documents, persist_directory):
    """Build a Chroma vector database from the given documents."""
    print(f"DEBUG: Attempting to build vector DB in directory: {persist_directory}")
    if not os.path.exists(persist_directory):
        print(f"DEBUG: Persist directory does not exist, creating: {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=persist_directory,
            client_settings=Settings(anonymized_telemetry=False)
        )
        print(f"DEBUG: Vector database built successfully in {persist_directory}")
        return vectorstore
    except Exception as e:
        print(f"ERROR: Detailed error building vector database: {e}")
        raise Exception(f"Error building vector database: {str(e)}")


def save_vector_db(vectorstore):
    """Persist Chroma vector database to disk."""
    try:
        vectorstore.persist()
    except Exception as e:
        raise Exception(f"Error saving vector database: {str(e)}")


def load_vector_db(persist_directory):
    """Load Chroma vector database from disk."""
    print(f"DEBUG: Attempting to load vector DB from directory: {persist_directory}")
    if not os.path.exists(persist_directory):
        print(f"DEBUG: Persist directory does not exist: {persist_directory}")
        raise Exception(f"Persist directory not found: {persist_directory}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=Settings(anonymized_telemetry=False)
        )
        print(f"DEBUG: Vector database loaded successfully from {persist_directory}")
        return vectorstore
    except Exception as e:
        print(f"ERROR: Detailed error loading vector database: {e}")
        raise Exception(f"Error loading vector database: {str(e)}")


def get_source_documents(vectorstore):
    """Get unique source document names from the vector database."""
    if not vectorstore:
        return []
    
    try:
        # This is a simplified way to get sources. For a production system,
        # you might store metadata more explicitly.
        all_docs = vectorstore.get()
        sources = [doc.get("metadata", {}).get("source") for doc in all_docs.get("metadatas", [])]
        return sorted(list(set(s for s in sources if s)))
    except Exception as e:
        # Handle cases where the vectorstore might not have a `get` method
        # or the metadata structure is different.
        print(f"Could not retrieve source documents: {e}")
        return []


def add_to_vector_db(vectorstore, documents):
    """Add new documents to an existing Chroma vector database."""
    try:
        vectorstore.add_documents(documents)
    except Exception as e:
        raise Exception(f"Error adding to vector database: {str(e)}")
