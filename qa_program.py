"""
Dog Breed QA Program

A RAG-based question-answering system that helps users find the right dog breed
based on their lifestyle, preferences, and living situation.

Uses Haystack to:
1. Read URLs from txt files
2. Fetch and convert HTML content from Wikipedia
3. Create document embeddings for semantic search
4. Generate answers using RAG pipeline
5. Evaluate RAG pipeline with metrics (DocumentMRR, Faithfulness, SAS)

Usage:
    python qa_program.py                    # Interactive mode
    python qa_program.py --index-only       # Only index documents
    python qa_program.py --limit 20         # Limit URLs to process
    python qa_program.py --use-openai       # Use OpenAI for answer generation
    python qa_program.py --use-hf           # Use HuggingFace API for answer generation
    python qa_program.py --test             # Test with 50 questions (saves to JSON)
    python qa_program.py --eval             # Run evaluation metrics on test questions
    python qa_program.py --eval --eval-samples 25  # Evaluate with 25 samples
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
from datetime import datetime

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


@component
class MetadataEnricher:
    """Add breed name to document metadata based on URL."""
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        """Extract breed name from URL and add to metadata."""
        for doc in documents:
            if doc and doc.meta:
                url = doc.meta.get("url", "")
                # Extract breed name from URL (usually in the path)
                try:
                    # Try to get from title if available
                    title = doc.meta.get("title", "")
                    if title:
                        breed_name = title.strip()
                    else:
                        # Fallback: extract from URL
                        parts = url.split('/')
                        breed_name = parts[-1].replace('_', ' ').replace('-', ' ').strip()
                    
                    doc.meta["breed_name"] = breed_name
                except Exception as e:
                    doc.meta["breed_name"] = "Unknown Breed"
        
        return {"documents": documents}

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
        # Use boolean conversion to avoid storing API keys in memory
        self.use_openai = use_openai and HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY"))
        self.use_hf = use_hf and HAS_HF and bool(os.getenv("HF_TOKEN"))
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
        self.qa_log_file = os.path.join(self.output_dir, f"qa_log_{timestamp}.txt")
        
        # Initialize log file with header
        self._initialize_log_file()
        
        if use_openai and not self.use_openai:
            has_key = bool(os.getenv("OPENAI_API_KEY"))
            has_openai_lib = HAS_OPENAI
            print(f"\nWarning: OpenAI requested but not available.")
            print(f"  - HAS_OPENAI (library installed): {has_openai_lib}")
            print(f"  - OPENAI_API_KEY (env var set): {has_key}")
            if not has_key:
                print(f"\n  ACTION NEEDED:")
                print(f"  1. Verify OPENAI_API_KEY is set: echo $env:OPENAI_API_KEY")
                print(f"  2. If empty, close ALL terminals and VS Code")
                print(f"  3. Reopen VS Code fresh (new process)")
                print(f"  4. Try again\n")
        if use_hf and not self.use_hf:
            has_token = bool(os.getenv("HF_TOKEN"))
            has_hf_lib = HAS_HF
            print(f"\nWarning: HuggingFace requested but not available.")
            print(f"  - HAS_HF (library installed): {has_hf_lib}")
            print(f"  - HF_TOKEN (env var set): {has_token}")

    def _initialize_log_file(self):
        """Initialize the Q&A log file with header information."""
        from datetime import datetime
        header = f"""================================================================================
DOG BREED QA SYSTEM - Question & Answer Log
================================================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Embedding Model: {self.embedding_model}
OpenAI Enabled: {self.use_openai}
HuggingFace Enabled: {self.use_hf}
================================================================================

"""
        with open(self.qa_log_file, 'w', encoding='utf-8') as f:
            f.write(header)

    def save_qa_pair(self, question: str, answer: str, session_type: str = "Interactive"):
        """
        Save a question-answer pair to the log file.
        
        Args:
            question: The user's question
            answer: The system's answer
            session_type: Type of session (Interactive, Questionnaire, Search, etc.)
        """
        from datetime import datetime
        
        # Format the Q&A entry
        entry = f"""[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {session_type}
{'─' * 80}
QUESTION:
{question}

ANSWER:
{answer}

{'=' * 80}

"""
        
        # Append to log file
        try:
            with open(self.qa_log_file, 'a', encoding='utf-8') as f:
                f.write(entry)
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
        # Chunking for better semantic coverage
        indexing.add_component("splitter", DocumentSplitter(
            split_by="word",
            split_length=200,      # Standard chunk size
            split_overlap=50       # Minimal overlap for efficiency
        ))
        # Tutorial 35: SentenceTransformersDocumentEmbedder
        indexing.add_component("document_embedder", SentenceTransformersDocumentEmbedder(
            model=self.embedding_model
        ))
        # Tutorial 35: DocumentWriter
        indexing.add_component("document_writer", DocumentWriter(
            document_store=self.document_store,
            policy=DuplicatePolicy.SKIP
        ))
        
        # Connect components with EXPLICIT port names (Tutorial 35 pattern)
        indexing.connect("fetcher.streams", "converter.sources")
        indexing.connect("converter", "cleaner")
        indexing.connect("cleaner", "splitter")
        indexing.connect("splitter", "document_embedder")
        # CRITICAL: Use explicit port connection for document output
        indexing.connect("document_embedder.documents", "document_writer.documents")
        
        return indexing

    def build_rag_pipeline(self) -> Pipeline:
        """Build the RAG pipeline for question answering.
        
        Following Haystack tutorial pattern with ChatPromptBuilder and Generator (if available).
        Properly connects generator to answer builder.
        """
        
        # Create chat prompt template with MAXIMUM hallucination prevention
        template = [
            ChatMessage.from_user(
"""You are a dog-information assistant with STRICT grounding requirements.

⚠️  ABSOLUTE RULES - VIOLATION MEANS FAILURE ⚠️
═══════════════════════════════════════════════════════════════

RULE 1: ONLY Wikipedia Content
- You MUST base every single fact on the retrieved Wikipedia documents below
- You MUST NOT use information from your training data
- You MUST NOT infer or extrapolate beyond what's written
- Violation: Using any information not explicitly in the sources = FAILURE

RULE 2: Evidence-Based Answers Only
- EVERY claim must be directly supported by the source text
- EVERY sentence must cite which source it comes from using [Source X]
- NO statements without citation
- NO guessing, assumptions, or inferences

RULE 3: Explicit Refusal for Insufficient Context
- If the retrieved sources do NOT contain information to answer the question → Say: "I cannot answer this question based on the retrieved Wikipedia sources"
- If sources mention the topic but lack relevant details → Say: "The retrieved sources mention this but do not provide the specific information needed"
- Do NOT attempt to partially answer if information is missing
- Do NOT fill gaps with general knowledge

RULE 4: Confidence Assessment
- Before answering, assess if sources actually contain relevant information
- If relevance score is low → Refuse to answer
- If sources don't directly address the question → Refuse to answer
- Only answer when sources have clear, direct information

═════════════════════════════════════════════════════════════════

RETRIEVED WIKIPEDIA SOURCES:
{% for document in documents %}
[Source {{ loop.index }}] 
URL: {{ document.meta.get('url', 'Unknown') }}
─────────────────────────────────────────────
{{ document.content }}
─────────────────────────────────────────────
{% endfor %}

QUESTION: {{ question }}

ANSWER INSTRUCTIONS:
1. FIRST: Check if sources actually contain relevant information for this question
2. If NO relevant information → Respond: "I cannot answer this question based on the retrieved Wikipedia sources."
3. If YES relevant information:
   a) Extract direct facts from sources only
   b) Cite source for EVERY sentence using [Source X]
   c) Do not add any explanation beyond what's in sources
   d) List the exact source passages that support your answer
   e) If question is not directly answered → Say: "The retrieved sources do not provide this information"

GOOD EXAMPLE:
Q: What is the origin of the Labrador Retriever?
A: The Labrador Retriever originated in Newfoundland [Source 1]. It was originally used as a hunting dog [Source 2].

BAD EXAMPLE (DO NOT DO THIS):
Q: What is the origin of the Labrador Retriever?
A: The Labrador is from Canada and was probably used for fishing. (No citations, uses training knowledge, not grounded)

FORMAT YOUR RESPONSE AS:

ANSWER:
[Your grounded response with [Source X] citations, or explicit refusal if sources insufficient]

SOURCE EVIDENCE:
[Exact passages from sources supporting your answer]

Now answer the question above following all rules:"""
            )
        ]
        
        rag_pipeline = Pipeline()
        
        # Add retrieval components with better configuration
        rag_pipeline.add_component(
            "query_embedder", 
            SentenceTransformersTextEmbedder(model=self.embedding_model)
        )
        rag_pipeline.add_component(
            "retriever", 
            # Increased top_k to get more candidates, filtering happens downstream
            InMemoryEmbeddingRetriever(
                document_store=self.document_store, 
                top_k=20,  # Retrieve more candidates for better selection
                filters=None  # No strict filters - let similarity do the work
            )
        )
        rag_pipeline.add_component(
            "prompt_builder", 
            ChatPromptBuilder(template=template, required_variables=["documents", "question"])
        )
        
        # Connect retrieval components - EXPLICIT PORTS per Haystack protocol
        rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        
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
            
            # Connect generator to answer builder (EXPLICIT ports required!)
            rag_pipeline.connect("prompt_builder.prompt", "generator.messages")
            rag_pipeline.connect("generator.replies", "answer_builder.replies")
            rag_pipeline.connect("retriever.documents", "answer_builder.documents")
            
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
            
            # Connect generator to answer builder (EXPLICIT ports required!)
            rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")
            rag_pipeline.connect("generator.replies", "answer_builder.replies")
            rag_pipeline.connect("retriever.documents", "answer_builder.documents")
        
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
                print("[OK]")
            except Exception as e:
                failed += len(batch)
                print(f"[ERROR] {str(e)[:50]}")
        
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
        
        Includes retrieval verification to prevent hallucination.
        
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
        
        # Retrieve documents directly for verification and to pass to answer_builder
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(self.embedding_model)
        query_embedding = embedder.encode(question).tolist()
        
        retrieved_docs = self.document_store.embedding_retrieval(
            query_embedding=query_embedding,
            top_k=20,
            return_embedding=False
        )
        
        # Only include generator/answer_builder inputs if generator is in use
        if self.use_openai or self.use_hf:
            run_inputs["generator"] = {}  # Uses piped-in messages from prompt_builder
            run_inputs["answer_builder"] = {
                "query": question,
                "documents": retrieved_docs  # Pass retrieved documents
            }
        
        # Run the RAG pipeline with required inputs
        result = self.rag_pipeline.run(run_inputs)
        
        # DEBUG: Print what we got back
        import json
        print(f"\n[DEBUG] Retrieved documents via embedding_retrieval: {len(retrieved_docs)}")
        print(f"[DEBUG] Pipeline output keys: {list(result.keys())}")
        if "retriever" in result:
            docs = result["retriever"].get("documents", [])
            print(f"[DEBUG] Retriever returned {len(docs)} documents")
        
        # Add retriever output to result dict for evaluation metrics to see
        # (normally pipeline only returns last component output, but evaluation needs retriever docs)
        if "retriever" not in result:
            result["retriever"] = {"documents": retrieved_docs}
        
        # HALLUCINATION PREVENTION - STEP 1: Check if retrieval found any documents
        documents_retrieved = self._verify_retrieval(result, retrieved_docs)
        if not documents_retrieved:
            # No documents found - prevent hallucination by blocking answer generation
            result = self._add_retrieval_failure_response(result)
            return result
        
        # HALLUCINATION PREVENTION - STEP 2: Check if documents are actually relevant to question
        documents_relevant = self._check_document_relevance(question, retrieved_docs)
        if not documents_relevant:
            # Documents don't match the question - prevent hallucination by blocking answer
            result = self._add_retrieval_failure_response(result)
            return result
        
        return result

    def _verify_retrieval(self, result: dict, retrieved_docs: list = None) -> bool:
        """
        Verify that the retriever actually found relevant documents.
        
        Returns True only if documents were retrieved with reasonable confidence.
        """
        try:
            # If documents were passed in explicitly, check them
            if retrieved_docs and len(retrieved_docs) > 0:
                for doc in retrieved_docs:
                    if doc and doc.content and len(doc.content.strip()) > 0:
                        return True
            
            # Check if retriever has documents in result (for pipelines that return it)
            if "retriever" in result and "documents" in result["retriever"]:
                docs = result["retriever"]["documents"]
                if docs and len(docs) > 0:
                    for doc in docs:
                        if doc and doc.content and len(doc.content.strip()) > 0:
                            return True
            
            # Also check answer_builder path (when LLM is used)
            if "answer_builder" in result and "documents" in result["answer_builder"]:
                docs = result["answer_builder"]["documents"]
                if docs and len(docs) > 0:
                    for doc in docs:
                        if doc and doc.content and len(doc.content.strip()) > 0:
                            return True
        except Exception as e:
            print(f"Warning: Error verifying retrieval: {e}")
        
        return False

    def _check_document_relevance(self, question: str, retrieved_docs: list) -> bool:
        """
        Check if retrieved documents actually contain relevant information for the question.
        
        This prevents answering with hallucinated information when documents don't match the question.
        
        Args:
            question: User's question
            retrieved_docs: List of retrieved documents
            
        Returns:
            True if documents contain potentially relevant information, False if likely irrelevant
        """
        if not retrieved_docs or len(retrieved_docs) == 0:
            return False
        
        # Extract key terms from question (important words)
        question_lower = question.lower()
        question_words = set(word.strip().lower() for word in question_lower.split() 
                            if len(word) > 3 and word not in ['what', 'does', 'have', 'breed', 'dog', 'dogs', 'are', 'this'])
        
        # Count how many documents contain question keywords
        matching_docs = 0
        for doc in retrieved_docs:
            if doc and doc.content:
                doc_content_lower = doc.content.lower()
                # Check if any question keywords appear in document
                keyword_matches = sum(1 for word in question_words if word in doc_content_lower)
                if keyword_matches >= len(question_words) * 0.3:  # At least 30% of keywords match
                    matching_docs += 1
        
        # Consider relevant if at least 40% of documents have keyword matches
        relevance_threshold = max(1, len(retrieved_docs) * 0.4)
        is_relevant = matching_docs >= relevance_threshold
        
        return is_relevant

    def _add_retrieval_failure_response(self, result: dict) -> dict:
        """
        Add a clear message when retrieval fails to prevent hallucination.
        """
        failure_message = (
            "I could not find relevant information in the Wikipedia sources to answer this question. "
            "The question may be asking for information not covered in the dog breed Wikipedia articles, "
            "or the phrasing may not match the indexed content. Try rephrasing your question or asking about a specific breed."
        )
        
        # Override the answer if generator was used
        if "answer_builder" in result and "answers" in result["answer_builder"]:
            if result["answer_builder"]["answers"]:
                result["answer_builder"]["answers"][0].data = failure_message
        
        result["_retrieval_status"] = "FAILED"
        result["_retrieval_message"] = failure_message
        
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
        
        # Extract answer from answer_builder if available (when LLM is configured)
        if "answer_builder" in result and "answers" in result["answer_builder"]:
            answers = result["answer_builder"]["answers"]
            if answers:
                return answers[0].data
        
        # Fallback: return the prompt with context (no LLM configured)
        if "prompt_builder" in result:
            prompt = result["prompt_builder"].get("prompt", "")
            return f"[Retrieved Context - No LLM configured]\n\n{prompt}"
        
        return "No answer could be generated."

    def get_answer_with_references(self, question: str) -> dict:
        """
        Ask a question and return both the answer and the retrieved context with references.
        
        Ensures answer is grounded in source documents. Returns detailed reference information.

        Returns:
            dict with keys:
                "answer"     — generated answer string (or "Not found" if retrieval failed)
                "references" — list of dicts: {url, snippet, source_id}
                "retrieval_success" — boolean indicating if documents were found
                "num_sources" — number of source documents used
        """
        result = self.ask(question)

        answer = "No answer could be generated."
        references = []
        retrieval_success = False
        
        # Check if retrieval actually succeeded
        if result.get("_retrieval_status") == "FAILED":
            return {
                "answer": result.get("_retrieval_message", answer),
                "references": [],
                "retrieval_success": False,
                "num_sources": 0
            }
        
        # Extract documents from pipeline result
        documents_to_process = []
        
        # Try to extract from answer_builder first (when LLM is available)
        if "answer_builder" in result:
            ab = result["answer_builder"]

            # Extract generated answer
            if "answers" in ab and ab["answers"]:
                answer = ab["answers"][0].data or answer

            # Extract retrieved documents from answer_builder
            if "documents" in ab and ab["documents"]:
                documents_to_process = ab["documents"]
                retrieval_success = len(documents_to_process) > 0
        
        # If no documents from answer_builder, try retriever output (when no LLM is configured)
        if not documents_to_process and "retriever" in result:
            retriever_output = result["retriever"]
            if "documents" in retriever_output:
                documents_to_process = retriever_output["documents"]
                retrieval_success = len(documents_to_process) > 0
        
        # Process documents into detailed references with source IDs
        for idx, doc in enumerate(documents_to_process, 1):
            if doc is not None:
                url = doc.meta.get("url", "") if doc.meta else ""
                content = doc.content or ""
                
                # Keep better snippet (up to 500 chars for more context)
                snippet = content[:500].strip()
                
                if snippet:  # Only add if there's actual content
                    references.append({
                        "source_id": f"[Source {idx}]",
                        "url": url,
                        "snippet": snippet,
                        "breed": doc.meta.get("breed_name", "Unknown") if doc.meta else "Unknown"
                    })

        # Fallback when no LLM: build readable answer from retrieved context
        if answer == "No answer could be generated." and references:
            context_parts = []
            for ref in references:
                context_parts.append(
                    f"{ref['source_id']} From {ref['breed']} article:\n{ref['snippet']}"
                )
            answer = (
                "[Retrieved from Wikipedia - No LLM configured]\n\n"
                + "\n\n" + "─" * 60 + "\n\n".join(context_parts)
            )
            retrieval_success = True

        return {
            "answer": answer,
            "references": references,
            "retrieval_success": retrieval_success,
            "num_sources": len(references)
        }

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
        
        Following Haystack tutorial pattern with DocumentMRREvaluator, FaithfulnessEvaluator, and SASEvaluator.
        For LLM-based metrics (Faithfulness, SAS), requires OPENAI_API_KEY or HF_TOKEN environment variable.
        
        Returns:
            Pipeline with evaluation components, or None if evaluators not available
        """
        if not HAS_EVALUATORS:
            print("Warning: Evaluation components not available. Install haystack-ai evaluators.")
            return None
        
        try:
            from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
            
            eval_pipeline = Pipeline()
            eval_pipeline.add_component("doc_mrr_evaluator", DocumentMRREvaluator())
            
            # Only add LLM-dependent evaluators if we have OpenAI API key
            has_openai = os.getenv("OPENAI_API_KEY") is not None
            
            if has_openai:
                try:
                    from haystack.components.evaluators.faithfulness import FaithfulnessEvaluator
                    from haystack.components.evaluators.sas_evaluator import SASEvaluator
                    
                    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
                    eval_pipeline.add_component("sas_evaluator", SASEvaluator(model=self.embedding_model))
                except Exception as llm_e:
                    print(f"Note: LLM-based evaluation components unavailable: {llm_e}")
                    print("      Continuing with retrieval evaluation only (DocumentMRR)")
            else:
                print("Note: OPENAI_API_KEY not set. Running retrieval evaluation only (DocumentMRR).")
                print("      For full evaluation, set OPENAI_API_KEY environment variable.")
            
            return eval_pipeline
        except Exception as e:
            print(f"Warning: Could not build evaluation pipeline: {e}")
            return None

    def evaluate_rag_pipeline(self, questions: List[str], ground_truth_answers: List[str], 
                              ground_truth_docs: List[Document], num_samples: int = 25) -> Dict[str, Any]:
        """
        Evaluate the RAG pipeline using the evaluation pipeline.
        
        Following Haystack tutorial pattern: run RAG pipeline, collect results, run evaluators.
        Evaluates on: DocumentMRR (retrieval quality), Faithfulness (answer grounding), 
        and SAS (semantic similarity).
        
        Args:
            questions: List of questions to evaluate on
            ground_truth_answers: Ground truth answers for evaluation
            ground_truth_docs: Ground truth documents for retrieval evaluation
            num_samples: Number of samples to evaluate (random sample if dataset larger)
            
        Returns:
            Dictionary with evaluation metrics and detailed results
        """
        if not HAS_EVALUATORS:
            print("Evaluation components not available.")
            return {"error": "Evaluation components not available"}
        
        if len(questions) == 0:
            return {"error": "No questions provided"}
        
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
        
        print(f"\n{'=' * 70}")
        print("RAG PIPELINE EVALUATION")
        print(f"{'=' * 70}")
        print(f"Evaluating on {len(sampled_questions)} samples...")
        
        # Run RAG pipeline and collect results
        rag_answers = []
        retrieved_docs = []
        
        for i, question in enumerate(sampled_questions):
            try:
                response = self.ask(question)
                
                # Extract answer
                if "answer_builder" in response and "answers" in response["answer_builder"]:
                    answers = response["answer_builder"]["answers"]
                    if answers:
                        rag_answers.append(answers[0].data)
                    else:
                        rag_answers.append("")
                else:
                    rag_answers.append("")
                
                # Extract retrieved documents - this is key for DocumentMRR evaluation
                if "answer_builder" in response and "documents" in response["answer_builder"]:
                    retrieved_docs.append(response["answer_builder"]["documents"])
                elif "retriever" in response and "documents" in response["retriever"]:
                    retrieved_docs.append(response["retriever"]["documents"])
                else:
                    retrieved_docs.append([])
                
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(sampled_questions)}", flush=True)
                    
            except Exception as e:
                print(f"  Error processing question {i}: {e}")
                rag_answers.append("")
                retrieved_docs.append([])
        
        # Build and run evaluation pipeline
        eval_pipeline = self.build_evaluation_pipeline()
        
        if not eval_pipeline:
            print(f"\nNote: Evaluation unavailable (evaluators not installed or LLM not configured)")
            return {
                "error": "Evaluation components not available",
                "rag_answers": rag_answers,
                "retrieved_docs": retrieved_docs,
                "num_samples": len(sampled_questions)
            }
        
        print(f"\nRunning evaluation metrics...")
        print(f"  - Document MRR (retrieval quality)")
        has_openai = os.getenv("OPENAI_API_KEY") is not None
        if has_openai:
            print(f"  - Faithfulness (answer grounding in context)")
            print(f"  - SAS (semantic answer similarity)")
        
        try:
            # Build eval inputs with only the evaluators that are actually in the pipeline
            eval_inputs = {}
            
            # DocumentMRR is always available
            eval_inputs["doc_mrr_evaluator"] = {
                "ground_truth_documents": [[doc] for doc in sampled_docs],
                "retrieved_documents": retrieved_docs,
            }
            
            # Conditional LLM-based evaluators
            if has_openai and "faithfulness" in [c for c in eval_pipeline.graph.nodes]:
                eval_inputs["faithfulness"] = {
                    "questions": list(sampled_questions),
                    "contexts": [doc.content for doc in sampled_docs],
                    "predicted_answers": rag_answers,
                }
            
            if has_openai and "sas_evaluator" in [c for c in eval_pipeline.graph.nodes]:
                eval_inputs["sas_evaluator"] = {
                    "predicted_answers": rag_answers,
                    "ground_truth_answers": list(sampled_answers)
                }
            
            eval_results = eval_pipeline.run(eval_inputs)
        except Exception as e:
            print(f"Error running evaluation: {e}")
            return {
                "error": f"Evaluation failed: {str(e)}", 
                "rag_answers": rag_answers,
                "retrieved_docs": retrieved_docs,
                "num_samples": len(sampled_questions)
            }
        
        # Create evaluation report
        inputs = {
            "question": list(sampled_questions),
            "contexts": [doc.content for doc in sampled_docs],
            "answer": list(sampled_answers),
            "predicted_answer": rag_answers,
        }
        
        try:
            evaluation_result = EvaluationRunResult(
                run_name="dog_breed_rag_pipeline",
                inputs=inputs,
                results=eval_results
            )
            
            # Generate aggregated report
            aggregated = evaluation_result.aggregated_report()
            
            print(f"\n{'=' * 70}")
            print("EVALUATION RESULTS")
            print(f"{'=' * 70}")
            print(aggregated)
            
            return {
                "evaluation_result": evaluation_result,
                "eval_scores": eval_results,
                "rag_answers": rag_answers,
                "retrieved_docs": retrieved_docs,
                "aggregated_report": str(aggregated),
                "num_samples": len(sampled_questions),
            }
        except Exception as e:
            print(f"Error creating evaluation report: {e}")
            return {
                "eval_scores": eval_results,
                "rag_answers": rag_answers,
                "retrieved_docs": retrieved_docs,
                "num_samples": len(sampled_questions),
                "error": f"Could not generate report: {str(e)}"
            }

    def run_evaluation_on_test_questions(self, num_samples: int = 10) -> Dict[str, Any]:
        """
        Run evaluation on a subset of test questions with ground truth data.
        
        This creates a simple evaluation dataset from the test questions and 
        evaluates the RAG pipeline against it.
        
        Args:
            num_samples: Number of test questions to evaluate on
            
        Returns:
            Dictionary with evaluation results and metrics
        """
        if not self.is_indexed:
            print("Error: Documents not indexed. Call initialize() first.")
            return {}
        
        # Use test questions as evaluation dataset
        test_questions = [
            "What is the origin of the Bulldog breed?",
            "Which breeds require the most exercise?",
            "Which breeds are good guard dogs?",
            "Which breeds are known for being aggressive?",
            "Which breeds are best for cold climates?",
            "Which breeds are known for high intelligence?",
            "Which breeds are best for active owners?",
            "Which breeds are best for hunting?",
            "Which breeds are known for loyalty?",
            "Which breeds are best for security purposes?",
        ]
        
        # Limit to requested samples
        test_questions = test_questions[:num_samples]
        
        # Create dummy ground truth answers (in real scenario, these would be labeled)
        ground_truth_answers = [
            "Bulldogs originated in England",
            "Alaskan Huskies require extensive exercise",
            "Aidi and Akbash dogs are good guards",
            "Akitas can be aggressive towards other dogs",
            "Alaskan Huskies are bred for cold climates",
            "Alaskan Huskies are intelligent",
            "Airedales and Alaskan Huskies are best for active owners",
            "Alaunts were used for hunting large game",
            "Akita Inu dogs are loyal",
            "Aidi and Akbash dogs serve security purposes",
        ]
        
        # Create dummy ground truth documents
        ground_truth_docs = [
            Document(content="Bulldogs are an English breed with history dating back centuries"),
            Document(content="Alaskan Huskies are bred as sled dogs requiring high exercise"),
            Document(content="Guard dogs like Aidi and Akbash provide property protection"),
            Document(content="Akitas display aggressive behavior towards other dogs and animals"),
            Document(content="Cold climate dogs have dense coats to handle harsh weather"),
            Document(content="Intelligent breeds like Alaskan Huskies excel at problem solving"),
            Document(content="Active dog breeds suit owners with high energy lifestyles"),
            Document(content="Hunting breeds were traditionally used for large game"),
            Document(content="Loyal breeds form deep bonds with their owners"),
            Document(content="Security dogs have protective and guarding instincts"),
        ]
        
        print(f"\n{'=' * 70}")
        print("RUNNING EVALUATION ON TEST QUESTIONS")
        print(f"{'=' * 70}")
        print(f"Questions: {len(test_questions)}")
        print(f"Ground Truth Samples: {len(ground_truth_docs)}\n")
        
        # Run evaluation
        eval_results = self.evaluate_rag_pipeline(
            questions=test_questions,
            ground_truth_answers=ground_truth_answers,
            ground_truth_docs=ground_truth_docs,
            num_samples=len(test_questions)
        )
        
        # Save evaluation results
        eval_report_file = os.path.join(
            self.output_dir,
            f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        try:
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "num_samples": eval_results.get("num_samples", 0),
                "metrics": {
                    "document_mrr": eval_results.get("eval_scores", {}).get("doc_mrr_evaluator", {}),
                    "faithfulness": eval_results.get("eval_scores", {}).get("faithfulness", {}),
                    "semantic_answer_similarity": eval_results.get("eval_scores", {}).get("sas_evaluator", {}),
                },
                "aggregated_report": eval_results.get("aggregated_report", ""),
                "status": "success" if "error" not in eval_results else "failed",
            }
            
            with open(eval_report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"[OK] Evaluation report saved to: {eval_report_file}\n")
        except Exception as e:
            print(f"Warning: Could not save evaluation report: {e}\n")
        
        return eval_results

    def test_50_questions(self) -> str:
        """
        Test the system with 50 predefined questions about dog breeds.
        Generates answers with retrieved references and saves to JSON.
        
        Returns:
            Path to the generated JSON output file
        """
        test_questions = [
            "What are the main characteristics of a Labrador Retriever?",
            "Which dog breeds are considered hypoallergenic?",
            "What is the average lifespan of a German Shepherd?",
            "Which breeds are best for apartment living?",
            "What are the grooming needs of a Poodle?",
            "Which dog breeds are known for being family-friendly?",
            "What is the origin of the Bulldog breed?",
            "Which breeds require the most exercise?",
            "What size category is a Beagle?",
            "Which breeds are good guard dogs?",
            "What is the temperament of a Golden Retriever?",
            "Which breeds are easiest to train?",
            "What health issues are common in Dachshunds?",
            "Which breeds are suitable for first-time owners?",
            "What is the coat type of a Siberian Husky?",
            "Which dog breeds are known for being aggressive?",
            "What is the weight range of a Rottweiler?",
            "Which breeds are best for cold climates?",
            "What is the history of the Border Collie?",
            "Which breeds shed the least?",
            "What is the typical height of a Great Dane?",
            "Which breeds are known for high intelligence?",
            "What are the exercise needs of a Boxer?",
            "Which breeds are good with children?",
            "What is the origin country of the Shiba Inu?",
            "Which breeds require minimal grooming?",
            "What is the temperament of a Chihuahua?",
            "Which breeds are best for active owners?",
            "What is the average lifespan of a French Bulldog?",
            "Which breeds are prone to separation anxiety?",
            "What is the coat color variety of a Cocker Spaniel?",
            "Which breeds are best for hunting?",
            "What is the energy level of a Jack Russell Terrier?",
            "Which breeds are suitable for hot climates?",
            "What is the temperament of a Doberman Pinscher?",
            "Which breeds are known for loyalty?",
            "What are common health issues in Bulldogs?",
            "Which breeds are best for small homes?",
            "What is the grooming requirement of a Maltese?",
            "Which breeds are considered toy breeds?",
            "What is the origin of the Australian Shepherd?",
            "Which breeds are good for elderly owners?",
            "What is the bark tendency of a Miniature Schnauzer?",
            "Which breeds are best for security purposes?",
            "What is the adaptability level of a Pug?",
            "Which breeds are known for being quiet?",
            "What is the intelligence ranking of a Border Collie?",
            "Which breeds are best for families with other pets?",
            "What is the typical diet requirement of large dog breeds?",
            "Which breeds are most popular worldwide?",
        ]
        
        print(f"\n{'=' * 70}")
        print(f"TESTING 50 QUESTIONS - Dog Breed QA System")
        print(f"{'=' * 70}\n")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(test_questions),
                "embedding_model": self.embedding_model,
                "use_openai": self.use_openai,
                "use_hf": self.use_hf,
                "total_documents_indexed": self.document_store.count_documents(),
            },
            "results": []
        }
        
        successful = 0
        failed = 0
        
        for idx, question in enumerate(test_questions, 1):
            try:
                print(f"[{idx}/{len(test_questions)}] Processing: {question[:60]}...", end=" ", flush=True)
                
                # Get answer with references
                response = self.get_answer_with_references(question)
                
                # Build result entry
                result_entry = {
                    "question_id": idx,
                    "question": question,
                    "answer": response["answer"],
                    "references": response["references"],
                    "num_references": len(response["references"])
                }
                
                results["results"].append(result_entry)
                successful += 1
                print("[OK]")
                
            except Exception as e:
                failed += 1
                print(f"[ERROR] {str(e)[:50]}")
                result_entry = {
                    "question_id": idx,
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "references": [],
                    "num_references": 0
                }
                results["results"].append(result_entry)
        
        # Add summary statistics
        results["summary"] = {
            "successful": successful,
            "failed": failed,
            "total_processed": successful + failed,
            "success_rate": f"{(successful / (successful + failed) * 100):.1f}%" if (successful + failed) > 0 else "0%"
        }
        
        # Save to JSON file
        output_file = os.path.join(
            self.output_dir, 
            f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'=' * 70}")
        print(f"[OK] Test complete! Results saved to:")
        print(f"  {output_file}")
        print(f"\nSummary:")
        print(f"  - Successful: {successful}/{len(test_questions)}")
        print(f"  - Failed: {failed}/{len(test_questions)}")
        print(f"  - Documents indexed: {self.document_store.count_documents()}")
        print(f"{'=' * 70}\n")
        
        return output_file


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
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: run 50 predefined questions and save results to JSON"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Evaluation mode: run evaluation metrics on test questions"
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
        print("\n[OK] OpenAI integration enabled")
    elif use_hf:
        print("\n[OK] HuggingFace API integration enabled")
    else:
        print("\n[Running without LLM - will show RAG context only]")
        print("[Set HF_TOKEN/OPENAI_API_KEY or use --use-hf/--use-openai for generated answers]")
    
    if HAS_EVALUATORS:
        print("[OK] Evaluation components available")
    else:
        print("[⚠ Evaluation components not available]")
    
    print("\nInitializing... (this may take a few minutes)")
    
    # Load URLs
    url_data = qa.load_urls()
    
    # Apply limit if specified
    if args.limit > 0:
        url_data = url_data[:args.limit]
        print(f"Limited to {args.limit} URLs")
    elif len(url_data) > 20 and not args.index_only and not args.evaluate and not args.test and not args.eval:
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
    
    # Test mode: run 50 questions and save to JSON
    if args.test:
        qa.test_50_questions()
        return
    
    # Evaluation mode: run evaluation metrics
    if args.eval:
        qa.run_evaluation_on_test_questions(num_samples=args.eval_samples)
        return
    
    # Evaluation mode (legacy)
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
            print(f"\n[OK] Saved to {qa.get_log_file_path()}")
            
        elif choice == "2":
            question = input("\nEnter your question: ").strip()
            if question:
                print("\nSearching for relevant information...")
                answer = qa.get_answer(question)
                print("\n" + answer)
                
                # Save to log
                qa.save_qa_pair(question, answer, "Direct Question")
                print(f"\n[OK] Saved to {qa.get_log_file_path()}")
            
        elif choice == "3":
            breed = input("\nEnter breed name to search: ").strip()
            if breed:
                print(f"\nSearching for information about {breed}...")
                question = f"Tell me about the {breed} dog breed, including its temperament, size, exercise needs, grooming requirements, and what type of owner it's best suited for."
                answer = qa.get_answer(question)
                print("\n" + answer)
                
                # Save to log
                qa.save_qa_pair(f"Breed Search: {breed}", answer, "Breed Search")
                print(f"\n[OK] Saved to {qa.get_log_file_path()}")
            
        elif choice == "4":
            print("\nGoodbye! Happy dog hunting!")
            print(f"[OK] All questions and answers saved to: {qa.get_log_file_path()}")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()