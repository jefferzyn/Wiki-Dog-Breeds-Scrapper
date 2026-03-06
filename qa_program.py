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
import argparse
import requests
import time
from pathlib import Path
from typing import List, Optional, Tuple
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
from haystack.components.builders import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import ByteStream

# Optional OpenAI integration
try:
    from haystack.components.generators import OpenAIGenerator
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Optional HuggingFace integration
try:
    from haystack.components.generators import HuggingFaceAPIGenerator
    HAS_HF = True
except ImportError:
    HAS_HF = False


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
QUESTIONNAIRE = [
    "1. What is your experience level with dogs? (beginner/intermediate/experienced)",
    "2. What type of home do you live in? (apartment/house with yard/rural property)",
    "3. How active is your lifestyle? (sedentary/moderate/very active)",
    "4. How many hours per day will the dog be left alone? (0-2/2-4/4-6/6+)",
    "5. Do you have children, and if so, what ages? (no children/toddlers/school-age/teenagers)",
    "6. Do you have other pets at home? (no/cats/dogs/other)",
    "7. How much time can you dedicate to daily exercise? (15-30min/30-60min/60+ min)",
    "8. How much time are you willing to spend on grooming? (minimal/moderate/extensive)",
    "9. Do you prefer a dog that is more independent or more affectionate/clingy? (independent/balanced/affectionate)",
    "10. Do you want a dog that is highly trainable and eager to please? (yes/no/doesn't matter)",
    "11. What size dog do you prefer? (small/medium/large/giant/no preference)",
    "12. Do you prefer a quiet dog or one that barks/vocalizes more? (quiet/moderate/vocal)",
    "13. What is the primary reason you want a dog? (companionship/protection/exercise partner/family pet/working)",
    "14. What climate do you live in? (hot/cold/temperate/varies)",
    "15. Do you prefer a puppy, young adult, adult, or senior dog? (puppy/young adult/adult/senior/no preference)",
    "16. Does anyone in your household have pet allergies? (yes/no)",
]


class DogBreedQA:
    """RAG-based QA system for dog breed recommendations."""

    def __init__(self, urls_dir: str = "data/urls", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_openai: bool = False,
                 use_hf: bool = False):
        """
        Initialize the QA system.
        
        Args:
            urls_dir: Directory containing txt files with Wikipedia URLs
            embedding_model: Sentence transformer model for embeddings
            use_openai: Whether to use OpenAI for answer generation
            use_hf: Whether to use HuggingFace API for answer generation
        """
        self.urls_dir = urls_dir
        self.embedding_model = embedding_model
        self.use_openai = use_openai and HAS_OPENAI and os.getenv("OPENAI_API_KEY")
        self.use_hf = use_hf and HAS_HF and os.getenv("HF_TOKEN")
        self.document_store = InMemoryDocumentStore()
        self.indexing_pipeline = None
        self.rag_pipeline = None
        self.is_indexed = False
        
        if use_openai and not self.use_openai:
            print("Warning: OpenAI requested but not available. Set OPENAI_API_KEY env var.")
        if use_hf and not self.use_hf:
            print("Warning: HuggingFace requested but not available. Set HF_TOKEN env var.")

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

    def build_indexing_pipeline(self) -> Pipeline:
        """Build the indexing pipeline for fetching and processing documents."""
        pipeline = Pipeline()
        
        # Add components - use custom fetcher with proper User-Agent for Wikipedia
        pipeline.add_component("fetcher", WikipediaFetcher(
            timeout=30,
            delay=0.3  # Be polite to Wikipedia
        ))
        pipeline.add_component("converter", HTMLToDocument())
        pipeline.add_component("cleaner", DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
        ))
        pipeline.add_component("splitter", DocumentSplitter(
            split_by="word",
            split_length=200,
            split_overlap=50
        ))
        pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(
            model=self.embedding_model
        ))
        pipeline.add_component("writer", DocumentWriter(
            document_store=self.document_store
        ))
        
        # Connect components
        pipeline.connect("fetcher.streams", "converter.sources")
        pipeline.connect("converter", "cleaner")
        pipeline.connect("cleaner", "splitter")
        pipeline.connect("splitter", "embedder")
        pipeline.connect("embedder", "writer")
        
        return pipeline

    def build_rag_pipeline(self) -> Pipeline:
        """Build the RAG pipeline for question answering."""
        
        prompt_template = """You are an expert dog breed advisor helping users find the perfect dog breed.

Based on the following Wikipedia information about various dog breeds, answer the user's question.

Context:
{% for document in documents %}
{{ document.content }}
---
{% endfor %}

User's Question: {{ question }}

Instructions:
- Provide specific breed recommendations when appropriate
- Consider size, temperament, exercise needs, grooming, trainability, and living situation compatibility
- Be honest if the provided context doesn't contain enough information
- Format your response clearly with breed names highlighted

Answer:"""
        
        pipeline = Pipeline()
        
        # Add components
        pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(
            model=self.embedding_model
        ))
        pipeline.add_component("retriever", InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=10
        ))
        pipeline.add_component("prompt_builder", PromptBuilder(
            template=prompt_template,
            required_variables=["documents", "question"]
        ))
        
        # Add LLM generator if available
        if self.use_openai:
            pipeline.add_component("generator", OpenAIGenerator(
                model="gpt-3.5-turbo",
                generation_kwargs={"max_tokens": 1000, "temperature": 0.7}
            ))
            pipeline.connect("prompt_builder", "generator")
        elif self.use_hf:
            pipeline.add_component("generator", HuggingFaceAPIGenerator(
                api_type="serverless_inference_api",
                api_params={"model": "mistralai/Mistral-7B-Instruct-v0.2"},
                generation_kwargs={"max_new_tokens": 500, "temperature": 0.7}
            ))
            pipeline.connect("prompt_builder", "generator")
        
        # Connect components
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever", "prompt_builder.documents")
        
        return pipeline

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
        
        print(f"Building indexing pipeline...")
        self.indexing_pipeline = self.build_indexing_pipeline()
        
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
        
        Args:
            question: User's question about dog breeds
            
        Returns:
            Dictionary with retrieved documents and generated answer/prompt
        """
        if not self.is_indexed:
            raise RuntimeError("Documents not indexed. Call initialize() first.")
        
        if self.rag_pipeline is None:
            self.rag_pipeline = self.build_rag_pipeline()
        
        # Run the RAG pipeline
        result = self.rag_pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question}
        })
        
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
        
        if (self.use_openai or self.use_hf) and "generator" in result:
            # Return LLM-generated answer
            replies = result.get("generator", {}).get("replies", [])
            if replies:
                return replies[0]
        
        # Return the prompt (context) for manual review
        prompt = result.get("prompt_builder", {}).get("prompt", "")
        return f"[Retrieved Context - No LLM configured]\n\n{prompt}"

    def interactive_questionnaire(self) -> str:
        """Run the interactive questionnaire and return compiled preferences."""
        print("\n" + "=" * 60)
        print("DOG BREED RECOMMENDATION QUESTIONNAIRE")
        print("=" * 60)
        print("\nPlease answer the following questions to help us find")
        print("the perfect dog breed for your lifestyle.\n")
        print("(Press Enter to skip any question)\n")
        
        answers = []
        for question in QUESTIONNAIRE:
            print(f"\n{question}")
            answer = input("Your answer: ").strip()
            if answer:
                answers.append(f"{question}\nAnswer: {answer}")
        
        if not answers:
            return "I'm looking for a dog breed recommendation."
        
        # Compile all answers into a single query
        compiled = "Based on my preferences, recommend suitable dog breeds:\n\n" + "\n\n".join(answers)
        return compiled


def create_simple_indexing_pipeline(document_store: InMemoryDocumentStore, 
                                    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Pipeline:
    """
    Create a simpler indexing pipeline that processes pre-fetched documents.
    
    This is useful when you want to manually control the fetching process
    or when working with already downloaded content.
    """
    pipeline = Pipeline()
    
    pipeline.add_component("cleaner", DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
    ))
    pipeline.add_component("splitter", DocumentSplitter(
        split_by="word",
        split_length=200,
        split_overlap=50
    ))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(
        model=embedding_model
    ))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    
    return pipeline


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
    return parser.parse_args()


def main():
    """Main entry point for the QA program."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("DOG BREED QA SYSTEM")
    print("=" * 60)
    
    # Initialize QA system
    use_openai = args.use_openai or bool(os.getenv("OPENAI_API_KEY"))
    use_hf = args.use_hf or bool(os.getenv("HF_TOKEN"))
    qa = DogBreedQA(urls_dir=args.urls_dir, use_openai=use_openai, use_hf=use_hf)
    
    print("\nThis system will help you find the perfect dog breed")
    print("based on your lifestyle and preferences.")
    
    if use_openai:
        print("\n[OpenAI integration enabled]")
    elif use_hf:
        print("\n[HuggingFace API integration enabled]")
    else:
        print("\n[Running without LLM - will show RAG context only]")
        print("[Set HF_TOKEN/OPENAI_API_KEY or use --use-hf/--use-openai for generated answers]")
    
    print("\nInitializing... (this may take a few minutes)")
    
    # Load URLs
    url_data = qa.load_urls()
    
    # Apply limit if specified
    if args.limit > 0:
        url_data = url_data[:args.limit]
        print(f"Limited to {args.limit} URLs")
    elif len(url_data) > 20 and not args.index_only:
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
            
        elif choice == "2":
            question = input("\nEnter your question: ").strip()
            if question:
                print("\nSearching for relevant information...")
                answer = qa.get_answer(question)
                print("\n" + answer)
            
        elif choice == "3":
            breed = input("\nEnter breed name to search: ").strip()
            if breed:
                print(f"\nSearching for information about {breed}...")
                question = f"Tell me about the {breed} dog breed, including its temperament, size, exercise needs, grooming requirements, and what type of owner it's best suited for."
                answer = qa.get_answer(question)
                print("\n" + answer)
            
        elif choice == "4":
            print("\nGoodbye! Happy dog hunting!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
