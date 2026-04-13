"""
Backend API Module for Dog Breed QA System

Provides a clean interface between Streamlit frontend and Haystack RAG pipeline.
Handles initialization, caching, and response formatting.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from qa_program import DogBreedQA


@dataclass
class QAResponse:
    """Response format for Q&A queries."""
    question: str
    answer: str
    retrieved_docs: List[Dict] = None
    relevance_score: float = 0.0
    is_confident: bool = True
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        """Convert response to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class QABackend:
    """Backend API for the Dog Breed QA System."""
    
    def __init__(self, urls_dir: str = "data/urls", 
                 output_dir: str = "data/qa_outputs",
                 use_openai: bool = False,
                 use_hf: bool = False):
        """
        Initialize the QA backend.
        
        Args:
            urls_dir: Directory containing breed URL files
            output_dir: Directory for saving logs
            use_openai: Enable OpenAI integration
            use_hf: Enable HuggingFace integration
        """
        self.qa_system = DogBreedQA(
            urls_dir=urls_dir,
            output_dir=output_dir,
            use_openai=use_openai,
            use_hf=use_hf
        )
        self.is_initialized = False
        self._init_status = "Not initialized"
    
    def initialize(self, limit: int = 0, url_data: Optional[List[tuple]] = None) -> Dict[str, any]:
        """
        Initialize the system by loading and indexing documents.
        
        Args:
            limit: Limit number of URLs to process (0 = all)
            url_data: Optional pre-loaded URL data
            
        Returns:
            Initialization status dictionary
        """
        try:
            if url_data is None:
                url_data = self.qa_system.load_urls()
            
            if limit > 0:
                url_data = url_data[:limit]
            
            self.qa_system.initialize(url_data)
            self.is_initialized = True
            self._init_status = "Ready"
            
            return {
                "status": "success",
                "message": "System initialized successfully",
                "urls_indexed": len(url_data),
                "documents": self.qa_system.document_store.count_documents(),
                "models": {
                    "embedding": self.qa_system.embedding_model,
                    "openai_enabled": self.qa_system.use_openai,
                    "huggingface_enabled": self.qa_system.use_hf
                }
            }
        except Exception as e:
            self._init_status = f"Error: {str(e)}"
            return {
                "status": "error",
                "message": str(e),
                "urls_indexed": 0
            }
    
    def answer_question(self, question: str, save_to_log: bool = True) -> QAResponse:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: The user's question
            save_to_log: Whether to save the Q&A pair to logs
            
        Returns:
            QAResponse object with answer and metadata
        """
        if not self.is_initialized:
            return QAResponse(
                question=question,
                answer="Error: System not initialized. Please run initialization first.",
                is_confident=False
            )
        
        try:
            from datetime import datetime
            
            # Get the answer
            answer = self.qa_system.get_answer(question)
            
            # Get raw response for metadata extraction
            raw_response = self.qa_system.ask(question)
            
            # Extract retrieved documents
            retrieved_docs = []
            if "retriever" in raw_response and "documents" in raw_response["retriever"]:
                for doc in raw_response["retriever"]["documents"]:
                    retrieved_docs.append({
                        "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "metadata": doc.meta if hasattr(doc, 'meta') else {}
                    })
            
            # Save to log if requested
            if save_to_log:
                self.qa_system.save_qa_pair(question, answer, "Streamlit Query")
            
            response = QAResponse(
                question=question,
                answer=answer,
                retrieved_docs=retrieved_docs,
                is_confident=True,
                timestamp=datetime.now().isoformat()
            )
            
            return response
            
        except Exception as e:
            return QAResponse(
                question=question,
                answer=f"Error processing question: {str(e)}",
                is_confident=False,
                timestamp=__import__('datetime').datetime.now().isoformat()
            )
    
    def answer_questionnaire(self, answers: Dict[str, str], save_to_log: bool = True) -> QAResponse:
        """
        Process questionnaire responses and get breed recommendations.
        
        Args:
            answers: Dictionary of questionnaire responses
            save_to_log: Whether to save to logs
            
        Returns:
            QAResponse with breed recommendations
        """
        if not self.is_initialized:
            return QAResponse(
                question="Questionnaire",
                answer="Error: System not initialized.",
                is_confident=False
            )
        
        try:
            from datetime import datetime
            
            # Format questionnaire responses as a question
            preferences_text = "\n".join([f"- {k}: {v}" for k, v in answers.items()])
            question = f"Based on the following preferences:\n{preferences_text}\n\nRecommend suitable dog breeds."
            
            # Get answer
            answer = self.qa_system.get_answer(question)
            
            # Save to log
            if save_to_log:
                self.qa_system.save_qa_pair(question, answer, "Questionnaire")
            
            return QAResponse(
                question="Breed Recommendations Questionnaire",
                answer=answer,
                is_confident=True,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return QAResponse(
                question="Questionnaire",
                answer=f"Error processing questionnaire: {str(e)}",
                is_confident=False,
                timestamp=__import__('datetime').datetime.now().isoformat()
            )
    
    def search_breed(self, breed_name: str, save_to_log: bool = True) -> QAResponse:
        """
        Search for information about a specific breed.
        
        Args:
            breed_name: Name of the breed
            save_to_log: Whether to save to logs
            
        Returns:
            QAResponse with breed information
        """
        question = f"Tell me about the {breed_name} dog breed, including its temperament, size, exercise needs, grooming requirements, and what type of owner it's best suited for."
        
        response = self.answer_question(question, save_to_log=False)
        response.question = f"Breed Search: {breed_name}"
        
        if save_to_log and response.is_confident:
            self.qa_system.save_qa_pair(response.question, response.answer, "Breed Search")
        
        return response
    
    def get_status(self) -> Dict[str, any]:
        """Get current system status."""
        status = {
            "initialized": self.is_initialized,
            "status": self._init_status,
            "documents_indexed": self.qa_system.document_store.count_documents() if self.is_initialized else 0,
            "embeddings_available": self.qa_system.use_openai or self.qa_system.use_hf,
            "log_file": self.qa_system.get_log_file_path()
        }
        return status
    
    def get_evaluators_status(self) -> Dict[str, bool]:
        """Check which evaluators are available."""
        from qa_program import HAS_EVALUATORS
        return {
            "evaluators_available": HAS_EVALUATORS,
            "faithfulness_evaluator": HAS_EVALUATORS,
            "sas_evaluator": HAS_EVALUATORS,
            "document_mrr_evaluator": HAS_EVALUATORS
        }
