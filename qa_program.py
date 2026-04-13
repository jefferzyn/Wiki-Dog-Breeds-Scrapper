"""
Dog Breed QA Program

A RAG-based question-answering system that helps users find the right dog breed
based on their lifestyle, preferences, and living situation.

Uses Haystack to:
1. Read URLs from txt files
2. Fetch and convert HTML content from Wikipedia
3. Create document embeddings for semantic search
4. Generate answers using RAG pipeline

Usage:
    python qa_program.py                    # Interactive mode
    python qa_program.py --index-only       # Only index documents
    python qa_program.py --limit 20         # Limit URLs to process
    python qa_program.py --use-openai       # Use OpenAI for answer generation
    python qa_program.py --use-hf           # Use HuggingFace API for answer generation
"""

import os
import sys
import glob
import json
import argparse
import requests
import time
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from haystack import Document, Pipeline, component
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.dataclasses import ChatMessage, ByteStream
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.evaluation.eval_run_result import EvaluationRunResult

# Optional OpenAI integration
try:
    from haystack.components.generators.chat import OpenAIChatGenerator
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Optional HuggingFace integration
try:
    from haystack.components.generators import HuggingFaceAPIGenerator
    HAS_HF = True
except ImportError:
    HAS_HF = False

# Evaluation components
try:
    from haystack.components.evaluators import (
        DocumentMRREvaluator,
        FaithfulnessEvaluator,
        SASEvaluator,
    )
    HAS_EVALUATORS = True
except ImportError:
    HAS_EVALUATORS = False


@component
class WikipediaFetcher:
    """Custom fetcher for Wikipedia with proper User-Agent headers."""
    
    def __init__(self, timeout: int = 30, delay: float = 0.5):
        self.timeout = timeout
        self.delay = delay  # Delay between requests to be polite
        self.headers = {
            "User-Agent": "DogBreedBot/1.0 (Educational project; https://github.com/example/dog-breeds) Python/3.x",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    
    @component.output_types(streams=List[ByteStream])
    def run(self, urls: List[str]) -> dict:
        """Fetch content from Wikipedia URLs with proper headers."""
        streams = []
        
        for url in urls:
            try:
                time.sleep(self.delay)  # Be polite to Wikipedia
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Create ByteStream from response content
                stream = ByteStream(
                    data=response.content,
                    meta={"url": url, "content_type": response.headers.get("content-type", "text/html")}
                )
                streams.append(stream)
                
            except requests.RequestException as e:
                print(f"Error fetching {url}: {e}")
                continue
        
        return {"streams": streams}

# Questions for dog breed recommendation
#QUESTIONNAIRE = [
   # "1. What is your experience level with dogs? (beginner/intermediate/experienced)",
   # "2. What type of home do you live in? (apartment/house with yard/rural property)",
   # "3. How active is your lifestyle? (sedentary/moderate/very active)",
   # "4. How many hours per day will the dog be left alone? (0-2/2-4/4-6/6+)",
   # "5. Do you have children, and if so, what ages? (no children/toddlers/school-age/teenagers)",
   # "6. Do you have other pets at home? (no/cats/dogs/other)",
   # "7. How much time can you dedicate to daily exercise? (15-30min/30-60min/60+ min)",
   # "8. How much time are you willing to spend on grooming? (minimal/moderate/extensive)",
   # "9. Do you prefer a dog that is more independent or more affectionate/clingy? (independent/balanced/affectionate)",
   # "10. Do you want a dog that is highly trainable and eager to please? (yes/no/doesn't matter)",
   # "11. What size dog do you prefer? (small/medium/large/giant/no preference)",
   # "12. Do you prefer a quiet dog or one that barks/vocalizes more? (quiet/moderate/vocal)",
   # "13. What is the primary reason you want a dog? (companionship/protection/exercise partner/family pet/working)",
   # "14. What climate do you live in? (hot/cold/temperate/varies)",
   # "15. Do you prefer a puppy, young adult, adult, or senior dog? (puppy/young adult/adult/senior/no preference)",
   # "16. Does anyone in your household have pet allergies? (yes/no)",
#]


class DogBreedQA:
    """RAG-based QA system for dog breed recommendations."""

    def __init__(self, urls_dir: str = "data/urls", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_openai: bool = False,
                 use_hf: bool = False,
                 output_dir: str = "data/qa_outputs"):
        """
        Initialize the QA system.
        
        Args:
            urls_dir: Directory containing txt files with Wikipedia URLs
            embedding_model: Sentence transformer model for embeddings
            use_openai: Whether to use OpenAI for answer generation
            use_hf: Whether to use HuggingFace API for answer generation
            output_dir: Directory to save Q&A outputs
        """
        self.urls_dir = urls_dir
        self.embedding_model = embedding_model
        self.use_openai = use_openai and HAS_OPENAI and os.getenv("OPENAI_API_KEY")
        self.use_hf = use_hf and HAS_HF and os.getenv("HF_TOKEN")
        self.document_store = InMemoryDocumentStore()
        self.indexing_pipeline = self._build_indexing_pipeline()  # Built once during init
        self.rag_pipeline = None
        self.is_indexed = False
        
        # Setup output directory for Q&A logging
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create timestamped log file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.qa_log_file = os.path.join(self.output_dir, f"qa_log_{timestamp}.json")
        
        # Initialize log file with header
        self._initialize_log_file()
        
        if use_openai and not self.use_openai:
            print("Warning: OpenAI requested but not available. Set OPENAI_API_KEY env var.")
        if use_hf and not self.use_hf:
            print("Warning: HuggingFace requested but not available. Set HF_TOKEN env var.")

    def _initialize_log_file(self):
        """Initialize the Q&A log file with JSON structure."""
        from datetime import datetime
        log_data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "embedding_model": self.embedding_model,
                "openai_enabled": self.use_openai,
                "huggingface_enabled": self.use_hf
            },
            "qa_pairs": []
        }
        with open(self.qa_log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def save_qa_pair(self, question: str, answer: str, session_type: str = "Interactive"):
        """
        Save a question-answer pair to the JSON log file.
        
        Args:
            question: The user's question
            answer: The system's answer
            session_type: Type of session (Interactive, Questionnaire, Search, etc.)
        """
        from datetime import datetime
        
        try:
            # Read existing data
            with open(self.qa_log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # Create new QA entry
            qa_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_type": session_type,
                "question": question,
                "answer": answer
            }
            
            # Add to the qa_pairs list
            log_data["qa_pairs"].append(qa_entry)
            
            # Write updated data back
            with open(self.qa_log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save Q&A pair to log: {e}")

    def get_log_file_path(self) -> str:
        """Get the path to the current Q&A log file."""
        return self.qa_log_file

    def load_urls(self) -> List[Tuple[str, str]]:
        """Load all URLs from txt files in the urls directory."""
        urls = []
        url_files = glob.glob(os.path.join(self.urls_dir, "*.txt"))
        
        print(f"Found {len(url_files)} URL files")
        
        for file_path in sorted(url_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    url = f.read().strip()
                    if url.startswith('http'):
                        breed_name = Path(file_path).stem
                        urls.append((url, breed_name))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        print(f"Loaded {len(urls)} URLs")
        return urls

    def _build_indexing_pipeline(self) -> Pipeline:
        """Build the indexing pipeline for fetching and processing documents.
        
        Following Haystack tutorial pattern for RAG pipeline evaluation.
        Built once during initialization to avoid redundant rebuilding.
        """
        indexing = Pipeline()
        
        # Add components - use custom fetcher with proper User-Agent for Wikipedia
        indexing.add_component("fetcher", WikipediaFetcher(
            timeout=30,
            delay=0.3  # Be polite to Wikipedia
        ))
        indexing.add_component("converter", HTMLToDocument())
        indexing.add_component("cleaner", DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
        ))
        indexing.add_component("splitter", DocumentSplitter(
            split_by="word",
            split_length=200,
            split_overlap=50
        ))
        indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(
            model=self.embedding_model
        ))
        indexing.add_component("writer", DocumentWriter(
            document_store=self.document_store,
            policy=DuplicatePolicy.SKIP
        ))
        
        # Connect components
        indexing.connect("fetcher.streams", "converter.sources")
        indexing.connect("converter", "cleaner")
        indexing.connect("cleaner", "splitter")
        indexing.connect("splitter", "embedder")
        indexing.connect("embedder", "writer")
        
        return indexing

    def build_rag_pipeline(self) -> Pipeline:
        """Build the RAG pipeline for question answering.
        
        Following Haystack tutorial pattern with ChatPromptBuilder and Generator (if available).
        Properly connects generator to answer builder.
        """
        
        # Create chat prompt template following tutorial pattern
        template = [
            ChatMessage.from_user(
"""You are a dog-information assistant answering questions about dogs and dog breeds.

Use ONLY the retrieved Wikipedia context below to answer the user's question.

Retrieved Context:
{% for document in documents %}
[Document {{ loop.index }}]
{{ document.content }}
---
{% endfor %}

User Question:
{{ question }}

Rules:
1. Base your answer only on the retrieved context.
2. Do not add facts that are not supported by the context. Do not guess or infer beyond what is explicitly stated in the context.
3. Every key claim in the answer must be supported by at least one citation [Document X].
4. If the context is insufficient, say clearly that the answer cannot be fully determined from the retrieved Wikipedia passages.
5. If the question asks for recommendations, suggestions, or comparisons, base them strictly on the retrieved context.
6. If the question is about choosing a breed, consider traits mentioned in the context such as size, temperament, activity level, grooming needs, trainability, and suitability for living situations.
7. If multiple documents are relevant, combine them into a coherent answer.
8. After the answer, provide a "References" section containing relevant excerpts from the retrieved context.
9. Only include references that directly support the answer.
10. Avoid repeating similar or redundant references.

Required Output Format:

Answer:
<clear answer with citations>

References:
- [Document X]: <relevant excerpt from passage>
- [Document Y]: <relevant excerpt from passage>
"""
            )
        ]
        
        rag_pipeline = Pipeline()
        
        # Add retrieval components
        rag_pipeline.add_component(
            "query_embedder", 
            SentenceTransformersTextEmbedder(model=self.embedding_model)
        )
        rag_pipeline.add_component(
            "retriever", 
            InMemoryEmbeddingRetriever(document_store=self.document_store, top_k=10)
        )
        rag_pipeline.add_component(
            "prompt_builder", 
            ChatPromptBuilder(template=template, required_variables=["documents", "question"])
        )
        
        # Connect retrieval components
        rag_pipeline.connect("query_embedder", "retriever.query_embedding")
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        
        # Add and connect generator if available
        if self.use_openai:
            from haystack.components.generators.chat import OpenAIChatGenerator
            rag_pipeline.add_component(
                "generator", 
                OpenAIChatGenerator(
                    model="gpt-4o-mini",
                    generation_kwargs={"max_tokens": 1000, "temperature": 0.7}
                )
            )
            rag_pipeline.add_component("answer_builder", AnswerBuilder())
            
            # Connect generator to answer builder
            rag_pipeline.connect("prompt_builder.prompt", "generator.messages")
            rag_pipeline.connect("generator.replies", "answer_builder.replies")
            rag_pipeline.connect("retriever", "answer_builder.documents")
            
        elif self.use_hf:
            rag_pipeline.add_component(
                "generator",
                HuggingFaceAPIGenerator(
                    api_type="serverless_inference_api",
                    api_params={"model": "mistralai/Mistral-7B-Instruct-v0.2"},
                    generation_kwargs={"max_new_tokens": 500, "temperature": 0.7}
                )
            )
            rag_pipeline.add_component("answer_builder", AnswerBuilder())
            
            # Connect generator to answer builder
            rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")
            rag_pipeline.connect("generator.replies", "answer_builder.replies")
            rag_pipeline.connect("retriever", "answer_builder.documents")
        else:
            # No generator: don't use answer_builder, just return documents
            # The ask() method will format documents as context
            pass
        
        return rag_pipeline

    def index_documents(self, url_data: Optional[List[tuple]] = None, batch_size: int = 5):
        """
        Index documents from Wikipedia URLs.
        
        Args:
            url_data: List of (url, breed_name) tuples to index
            batch_size: Number of URLs to process at once
        """
        if url_data is None:
            url_data = self.load_urls()
        
        if not url_data:
            print("No URLs to index!")
            return
        
        urls = [item[0] if isinstance(item, tuple) else item for item in url_data]
        
        total_batches = (len(urls) + batch_size - 1) // batch_size
        print(f"Indexing {len(urls)} URLs in {total_batches} batches...")
        
        successful = 0
        failed = 0
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_num = i // batch_size + 1
            try:
                print(f"Processing batch {batch_num}/{total_batches}...", end=" ", flush=True)
                self.indexing_pipeline.run({"fetcher": {"urls": batch}})
                successful += len(batch)
                print("✓")
            except Exception as e:
                failed += len(batch)
                print(f"✗ Error: {str(e)[:50]}")
        
        doc_count = self.document_store.count_documents()
        print(f"\nIndexing complete:")
        print(f"  - URLs processed: {successful} successful, {failed} failed")
        print(f"  - Document chunks created: {doc_count}")
        self.is_indexed = True

    def initialize(self, url_data: Optional[List[tuple]] = None):
        """Initialize the QA system by indexing documents and building RAG pipeline."""
        self.index_documents(url_data)
        print("Building RAG pipeline...")
        self.rag_pipeline = self.build_rag_pipeline()
        print("QA system ready!")

    def ask(self, question: str) -> dict:
        """
        Ask a question and get an answer.
        
        Following tutorial pattern - run with all required components.
        
        Args:
            question: User's question about dog breeds
            
        Returns:
            Dictionary with retrieved documents and generated answer
        """
        if not self.is_indexed:
            raise RuntimeError("Documents not indexed. Call initialize() first.")
        
        if self.rag_pipeline is None:
            self.rag_pipeline = self.build_rag_pipeline()
        
        # Build inputs based on what's in the pipeline
        run_inputs = {
            "query_embedder": {"text": question},
            "prompt_builder": {"question": question},
        }
        
        # Only include answer_builder and generator if generator is in use
        if self.use_openai or self.use_hf:
            run_inputs["answer_builder"] = {"query": question}
            run_inputs["generator"] = {}  # Uses piped-in messages from prompt_builder
        
        # Run the RAG pipeline with required inputs
        result = self.rag_pipeline.run(run_inputs)
        
        return result

    def get_answer(self, question: str) -> str:
        """
        Get a formatted answer to a question.
        
        Args:
            question: User's question
            
        Returns:
            Formatted answer string
        """
        result = self.ask(question)
        
        # Extract answer from answer_builder if using generator
        if self.use_openai or self.use_hf:
            if "answer_builder" in result and "answers" in result["answer_builder"]:
                answers = result["answer_builder"]["answers"]
                if answers:
                    return answers[0].data
        
        # Context-only mode: format retrieved documents
        if "retriever" in result and "documents" in result["retriever"]:
            documents = result["retriever"]["documents"]
            if documents:
                formatted_answer = "Based on Wikipedia articles about dog breeds:\n\n"
                for i, doc in enumerate(documents[:5], 1):
                    content = doc.content[:400] if len(doc.content) > 400 else doc.content
                    formatted_answer += f"[Document {i}]\n{content}\n\n"
                return formatted_answer
        
        return "No answer could be generated. Please ensure documents are indexed."

    def interactive_questionnaire(self) -> str:
        """Run the interactive questionnaire and return compiled preferences."""
        print("\n" + "=" * 60)
        print("DOG BREED RECOMMENDATION QUESTIONNAIRE")
        print("=" * 60)
        print("\nPlease answer the following questions to help us find")
        print("the perfect dog breed for your lifestyle.\n")
        print("(Press Enter to skip any question)\n")
        
        questionnaire = [
            "1. What are the main characteristics of a Labrador Retriever?",
            "2. Which dog breeds are considered hypoallergenic?",
            "3. What is the average lifespan of a German Shepherd?",
            "4. Which breeds are best for apartment living?",
            "5. What are the grooming needs of a Poodle?",
            "6. Which dog breeds are known for being family-friendly?",
            "7. What is the origin of the Bulldog breed?",
            "8. Which breeds require the most exercise?",
            "9. What size category is a Beagle?",
            "10. Which breeds are good guard dogs?",
            "11. What is the temperament of a Golden Retriever?",
            "12. Which breeds are easiest to train?",
            "13. What health issues are common in Dachshunds?",
            "14. Which breeds are suitable for first-time owners?",
            "15. What is the coat type of a Siberian Husky?",
            "16. Which breeds are known for being aggressive?",
            "17. What is the weight range of a Rottweiler?",
            "18. Which breeds are best for cold climates?",
            "19. What is the history of the Border Collie?",
            "20. Which breeds shed the least?",
            "21. What is the typical height of a Great Dane?",
            "22. Which breeds are known for high intelligence?",
            "23. What are the exercise needs of a Boxer?",
            "24. Which breeds are good with children?",
            "25. What is the origin country of the Shiba Inu?",
            "26. Which breeds require minimal grooming?",
            "27. What is the temperament of a Chihuahua?",
            "28. Which breeds are best for active owners?",
            "29. What is the average lifespan of a French Bulldog?",
            "30. Which breeds are prone to separation anxiety?",
            "31. What is the coat color variety of a Cocker Spaniel?",
            "32. Which breeds are best for hunting?",
            "33. What is the energy level of a Jack Russell Terrier?",
            "34. Which breeds are suitable for hot climates?",
            "35. What is the temperament of a Doberman Pinscher?",
            "36. Which breeds are known for loyalty?",
            "37. What are common health issues in Bulldogs?",
            "38. Which breeds are best for small homes?",
            "39. What is the grooming requirement of a Maltese?",
            "40. Which breeds are considered toy breeds?",
            "41. What is the origin of the Australian Shepherd?",
            "42. Which breeds are good for elderly owners?",
            "43. What is the bark tendency of a Miniature Schnauzer?",
            "44. Which breeds are best for security purposes?",
            "45. What is the adaptability level of a Pug?",
            "46. Which breeds are known for being quiet?",
            "47. What is the intelligence ranking of a Border Collie?",
            "48. Which breeds are best for families with other pets?",
            "49. What is the typical diet requirement of large dog breeds?",
            "50. Which breeds are most popular worldwide?",
        ]
        
        answers = []
        for question in questionnaire:
            print(f"\n{question}")
            answer = input("Your answer: ").strip()
            if answer:
                answers.append(f"{question}\nAnswer: {answer}")
        
        if not answers:
            return "I'm looking for a dog breed recommendation."
        
        # Compile all answers into a single query
        compiled = "Based on my preferences, recommend suitable dog breeds:\n\n" + "\n\n".join(answers)
        return compiled

    def build_evaluation_pipeline(self) -> Pipeline:
        """Build evaluation pipeline for RAG pipeline assessment.
        
        Following tutorial pattern with DocumentMRREvaluator, FaithfulnessEvaluator, and SASEvaluator.
        """
        if not HAS_EVALUATORS:
            print("Warning: Evaluation components not available. Install haystack-ai evaluators.")
            return None
        
        eval_pipeline = Pipeline()
        eval_pipeline.add_component("doc_mrr_evaluator", DocumentMRREvaluator())
        eval_pipeline.add_component(
            "sas_evaluator", 
            SASEvaluator(model=self.embedding_model)
        )
        eval_pipeline.add_component("faithfulness_evaluator", FaithfulnessEvaluator())
        
        return eval_pipeline

    def evaluate_rag_pipeline(self, questions: List[str], ground_truth_answers: List[str], 
                              ground_truth_docs: List[Document], num_samples: int = 25) -> Dict[str, Any]:
        """
        Evaluate the RAG pipeline using the evaluation pipeline.
        
        Following tutorial pattern: run RAG pipeline, collect results, run evaluators.
        
        Args:
            questions: List of questions to evaluate on
            ground_truth_answers: Ground truth answers for evaluation
            ground_truth_docs: Ground truth documents for retrieval evaluation
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation results dictionary with metrics
        """
        if not HAS_EVALUATORS:
            print("Evaluation components not available.")
            return {}
        
        # Sample data if too large
        if len(questions) > num_samples:
            indices = random.sample(range(len(questions)), num_samples)
            sampled_questions = [questions[i] for i in indices]
            sampled_answers = [ground_truth_answers[i] for i in indices]
            sampled_docs = [ground_truth_docs[i] for i in indices]
        else:
            sampled_questions = questions
            sampled_answers = ground_truth_answers
            sampled_docs = ground_truth_docs
        
        print(f"\nEvaluating RAG pipeline on {len(sampled_questions)} samples...")
        
        # Run RAG pipeline and collect results
        rag_answers = []
        retrieved_docs = []
        
        for i, question in enumerate(sampled_questions):
            try:
                response = self.ask(question)
                
                # Extract answer from answer_builder (if using generator)
                if "answer_builder" in response and "answers" in response["answer_builder"]:
                    answers = response["answer_builder"]["answers"]
                    if answers:
                        rag_answers.append(answers[0].data)
                    else:
                        rag_answers.append("")
                else:
                    rag_answers.append("")
                
                # Extract retrieved documents (from retriever component)
                if "retriever" in response and "documents" in response["retriever"]:
                    retrieved_docs.append(response["retriever"]["documents"])
                else:
                    retrieved_docs.append([])
                
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(sampled_questions)}")
                    
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                rag_answers.append("")
                retrieved_docs.append([])
        
        # Build and run evaluation pipeline
        eval_pipeline = self.build_evaluation_pipeline()
        
        print("\nRunning evaluators...")
        eval_results = eval_pipeline.run({
            "doc_mrr_evaluator": {
                "ground_truth_documents": [[doc] for doc in sampled_docs],
                "retrieved_documents": retrieved_docs,
            },
            "sas_evaluator": {
                "predicted_answers": rag_answers,
                "ground_truth_answers": sampled_answers,
            },
            "faithfulness_evaluator": {
                "questions": sampled_questions,
                "contexts": [doc.content for doc in sampled_docs],
                "predicted_answers": rag_answers,
            },
        })
        
        # Create evaluation report
        inputs = {
            "question": sampled_questions,
            "contexts": [doc.content for doc in sampled_docs],
            "answer": sampled_answers,
            "predicted_answer": rag_answers,
        }
        
        evaluation_result = EvaluationRunResult(
            run_name="dog_breed_rag_pipeline",
            inputs=inputs,
            results=eval_results
        )
        
        return {
            "evaluation_result": evaluation_result,
            "eval_scores": eval_results,
            "rag_answers": rag_answers,
            "retrieved_docs": retrieved_docs,
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dog Breed QA System - Find the perfect dog breed for your lifestyle"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=0,
        help="Limit number of URLs to process (0 = all)"
    )
    parser.add_argument(
        "--index-only", action="store_true",
        help="Only index documents, don't run interactive mode"
    )
    parser.add_argument(
        "--use-openai", action="store_true",
        help="Use OpenAI for answer generation (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--use-hf", action="store_true",
        help="Use HuggingFace API for answer generation (requires HF_TOKEN)"
    )
    parser.add_argument(
        "--urls-dir", type=str, default="data/urls",
        help="Directory containing URL files"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=5,
        help="Batch size for indexing URLs"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation mode (requires ground truth data)"
    )
    parser.add_argument(
        "--eval-samples", type=int, default=25,
        help="Number of samples to evaluate on"
    )
    return parser.parse_args()


def main():
    """Main entry point for the QA program."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("DOG BREED QA SYSTEM - Haystack RAG Pipeline")
    print("=" * 60)
    
    # Initialize QA system
    use_openai = args.use_openai or bool(os.getenv("OPENAI_API_KEY"))
    use_hf = args.use_hf or bool(os.getenv("HF_TOKEN"))
    qa = DogBreedQA(urls_dir=args.urls_dir, use_openai=use_openai, use_hf=use_hf)
    
    print("\nThis system helps you find the perfect dog breed")
    print("based on your lifestyle and preferences.")
    
    if use_openai:
        print("\n[✓ OpenAI integration enabled]")
    elif use_hf:
        print("\n[✓ HuggingFace API integration enabled]")
    else:
        print("\n[Running without LLM - will show RAG context only]")
        print("[Set HF_TOKEN/OPENAI_API_KEY or use --use-hf/--use-openai for generated answers]")
    
    if HAS_EVALUATORS:
        print("[✓ Evaluation components available]")
    else:
        print("[⚠ Evaluation components not available]")
    
    print("\nInitializing... (this may take a few minutes)")
    
    # Load URLs
    url_data = qa.load_urls()
    
    # Apply limit if specified
    if args.limit > 0:
        url_data = url_data[:args.limit]
        print(f"Limited to {args.limit} URLs")
    elif len(url_data) > 20 and not args.index_only and not args.evaluate:
        print(f"\nFound {len(url_data)} URLs. For faster testing, you can:")
        print("1. Process all URLs (may take 10-20 minutes)")
        print("2. Process first 20 URLs for quick demo")
        print("3. Process first 50 URLs")
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "2":
            url_data = url_data[:20]
        elif choice == "3":
            url_data = url_data[:50]
        print(f"Using {len(url_data)} URLs.")
    
    # Initialize with URLs
    qa.initialize(url_data)
    
    if args.index_only:
        print("\nIndexing complete. Exiting.")
        return
    
    # Evaluation mode
    if args.evaluate:
        print("\n" + "=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)
        print("This mode requires ground truth data (questions, answers, documents).")
        print("For now, running a demo evaluation on the indexed documents.")
        
        # Sample questions
        questions = [
            "Which dog breeds are best for apartment living?",
            "What breeds are good for first-time dog owners?",
            "Which breeds have the highest energy levels?",
            "What breeds are best with children?",
            "Which dog breeds require the least grooming?",
        ]
        
        print(f"\nNote: Full evaluation requires {args.eval_samples} ground truth Q&A pairs.")
        print("This demo uses limited sample questions.")
        return
    
    # Interactive mode
    while True:
        print("\n" + "-" * 40)
        print("OPTIONS:")
        print("1. Take questionnaire for breed recommendations")
        print("2. Ask a specific question about dog breeds")
        print("3. Search for a specific breed")
        print("4. Exit")
        print("-" * 40)
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            # Run questionnaire
            preferences = qa.interactive_questionnaire()
            print("\n" + "=" * 60)
            print("ANALYZING YOUR PREFERENCES...")
            print("=" * 60)
            
            answer = qa.get_answer(preferences)
            print("\n" + answer)
            
            # Save to log
            qa.save_qa_pair(preferences, answer, "Questionnaire")
            print(f"\n[✓ Saved to {qa.get_log_file_path()}]")
            
        elif choice == "2":
            question = input("\nEnter your question: ").strip()
            if question:
                print("\nSearching for relevant information...")
                answer = qa.get_answer(question)
                print("\n" + answer)
                
                # Save to log
                qa.save_qa_pair(question, answer, "Direct Question")
                print(f"\n[✓ Saved to {qa.get_log_file_path()}]")
            
        elif choice == "3":
            breed = input("\nEnter breed name to search: ").strip()
            if breed:
                print(f"\nSearching for information about {breed}...")
                question = f"Tell me about the {breed} dog breed, including its temperament, size, exercise needs, grooming requirements, and what type of owner it's best suited for."
                answer = qa.get_answer(question)
                print("\n" + answer)
                
                # Save to log
                qa.save_qa_pair(f"Breed Search: {breed}", answer, "Breed Search")
                print(f"\n[✓ Saved to {qa.get_log_file_path()}]")
            
        elif choice == "4":
            print("\nGoodbye! Happy dog hunting! 🐕")
            print(f"[✓ All questions and answers saved to: {qa.get_log_file_path()}]")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
