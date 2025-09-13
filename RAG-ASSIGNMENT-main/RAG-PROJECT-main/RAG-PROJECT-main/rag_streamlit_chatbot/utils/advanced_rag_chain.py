import os
import requests
import json
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from pydantic import Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRouterLLM(LLM):
    """OpenRouter LLM implementation."""
    model: str = "openai/gpt-3.5-turbo"  # Default OpenRouter model
    temperature: float = 0.1
    max_tokens: int = 1024
    api_key: str = Field(..., env="OPENROUTER_API_KEY")
    site_url: Optional[str] = None
    site_name: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "site_url": self.site_url,
            "site_name": self.site_name,
        }

def get_llm():
    """
    Get OpenRouter LLM instance.
    
    Returns:
        OpenRouterLLM: Configured LLM instance
    """
    try:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in your .env file.")
        
        return OpenRouterLLM(
            api_key=openrouter_api_key,
            model="openai/gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1024,
            site_url=os.getenv("OPENROUTER_SITE_URL"),
            site_name=os.getenv("OPENROUTER_SITE_NAME")
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

def create_advanced_rag_chain(vectorstore):
    """
    Create an advanced RAG chain with enhanced capabilities.
    
    Args:
        vectorstore: Vector store instance
        
    Returns:
        dict: Dictionary containing different RAG chains for different query types
    """
    try:
        # Get LLM
        llm = get_llm()
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # Increased to 8 for better context
        )

        # Enhanced prompt templates for different query types
        base_prompt_template = """
You are a helpful AI research assistant. Use the following context to answer the user's question accurately and comprehensively.

Context: {context}

Question: {question}

Instructions:
- Provide a clear, detailed answer based on the context
- If the information is not available in the context, provide a helpful response explaining what information is available
- Structure your response in a helpful, educational manner
- Use examples from the context when relevant
- Be specific and reference the source material when possible

Answer:"""

        summary_prompt_template = """
You are a helpful AI research assistant. Summarize the following research content clearly and concisely.

Context: {context}

Task: {question}

Instructions:
- Provide a clear, concise summary focusing on the key points
- Highlight key findings, methodology, or conclusions
- Maintain accuracy to the source material
- Structure your response logically
- Include relevant details from the research

Summary:"""

        extraction_prompt_template = """
You are a precise AI research assistant. Extract specific information from the following research content.

Context: {context}

Task: {question}

Instructions:
- Extract only the requested specific information
- If the exact information is not available, provide the closest relevant information
- Format your answer clearly and structured
- Include quantitative data when requested (accuracy, scores, metrics)
- Reference specific sections or findings when possible

Extracted Information:"""

        # Create prompt templates
        base_prompt = PromptTemplate(
            template=base_prompt_template,
            input_variables=["context", "question"]
        )
        
        summary_prompt = PromptTemplate(
            template=summary_prompt_template,
            input_variables=["context", "question"]
        )
        
        extraction_prompt = PromptTemplate(
            template=extraction_prompt_template,
            input_variables=["context", "question"]
        )

        # Create different RAG chains for different query types
        base_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": base_prompt}
        )
        
        summary_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": summary_prompt}
        )
        
        extraction_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": extraction_prompt}
        )

        # Return dictionary of chains
        return {
            "base": base_chain,
            "summary": summary_chain,
            "extraction": extraction_chain,
            "vectorstore": vectorstore,
            "retriever": retriever
        }
        
    except Exception as e:
        raise Exception(f"Error creating advanced RAG chain: {str(e)}")

def process_specialized_query(rag_chain_dict, query):
    """
    Process specialized queries with appropriate prompt templates.
    
    Args:
        rag_chain_dict: Dictionary containing different RAG chains
        query (str): User query
        
    Returns:
        dict: Query result with answer and source documents
    """
    try:
        # Determine query type and use appropriate chain
        query_lower = query.lower()
        
        # Enhanced query type detection with more keywords
        summary_keywords = ["summarize", "summary", "overview", "abstract", "main points", "key points"]
        extraction_keywords = ["accuracy", "f1", "score", "metric", "result", "performance", "percentage", "value"]
        conclusion_keywords = ["conclude", "conclusion", "finding", "outcome", "result", "implication"]
        methodology_keywords = ["methodology", "method", "approach", "technique", "algorithm", "procedure"]
        
        # Determine which chain to use
        if any(keyword in query_lower for keyword in summary_keywords):
            # Use summary chain
            logger.info(f"Using summary chain for query: {query}")
            result = rag_chain_dict["summary"]({"query": query})
        elif any(keyword in query_lower for keyword in extraction_keywords):
            # Use extraction chain for quantitative results
            logger.info(f"Using extraction chain for query: {query}")
            result = rag_chain_dict["extraction"]({"query": query})
        elif any(keyword in query_lower for keyword in conclusion_keywords):
            # Use extraction chain for conclusions
            logger.info(f"Using extraction chain for conclusion query: {query}")
            result = rag_chain_dict["extraction"]({"query": query})
        elif any(keyword in query_lower for keyword in methodology_keywords):
            # Use extraction chain for methodology
            logger.info(f"Using extraction chain for methodology query: {query}")
            result = rag_chain_dict["extraction"]({"query": query})
        else:
            # Use base chain for general queries
            logger.info(f"Using base chain for general query: {query}")
            result = rag_chain_dict["base"]({"query": query})
        
        # Post-process the result to improve response quality
        if "result" in result:
            # Remove apologetic language and improve tone
            result["result"] = result["result"].replace("I'm sorry", "Based on the available information")
            result["result"] = result["result"].replace("I apologize", "According to the research")
            result["result"] = result["result"].replace("I don't have specific information", "The available research indicates")
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing specialized query: {str(e)}")
        # Return a fallback response instead of raising an exception
        return {
            "result": f"I encountered an issue processing your query. Please try rephrasing your question or check if documents have been properly loaded. Error: {str(e)}",
            "source_documents": []
        }