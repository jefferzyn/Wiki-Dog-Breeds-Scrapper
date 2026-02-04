"""
Haystack pipeline for dog breed search and retrieval.
Indexes scraped dog breeds and enables semantic search.
"""

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from scraper import scrape_dog_breeds


class DogBreedPipeline:
    """Manages the Haystack pipeline for dog breed search."""

    def __init__(self):
        """Initialize the pipeline with document store."""
        self.document_store = InMemoryDocumentStore()
        self.retriever = InMemoryBM25Retriever(document_store=self.document_store)
        self.pipeline = None

    def load_data(self, fetch_descriptions: bool = False):
        """Scrape and load dog breed data into the document store."""
        print("Loading dog breed data...")
        breeds = scrape_dog_breeds(fetch_descriptions=fetch_descriptions)

        # Convert breed data to Haystack Documents
        documents = [
            Document(
                content=f"{breed['name']}. {breed.get('description', '')}".strip(),
                meta={
                    "breed_name": breed["name"],
                    "url": breed.get("url", ""),
                    "source": "wikipedia"
                },
            )
            for breed in breeds
        ]

        self.document_store.write_documents(documents)
        print(f"Loaded {len(documents)} documents into document store")

    def build_pipeline(self):
        """Build the search pipeline."""
        self.pipeline = Pipeline()
        self.pipeline.add_component("retriever", self.retriever)

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Search for dog breeds matching the query.

        Args:
            query: Search query (e.g., "small dog breeds")
            top_k: Number of top results to return

        Returns:
            List of matching documents
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")

        results = self.retriever.run(query=query, top_k=top_k)
        return results.get("documents", [])

    def initialize(self, fetch_descriptions: bool = False):
        """Initialize pipeline with data."""
        self.load_data(fetch_descriptions=fetch_descriptions)
        self.build_pipeline()


def create_pipeline(fetch_descriptions: bool = False) -> DogBreedPipeline:
    """Create and initialize a dog breed search pipeline."""
    pipeline = DogBreedPipeline()
    pipeline.initialize(fetch_descriptions=fetch_descriptions)
    return pipeline


if __name__ == "__main__":
    # Example usage
    pipeline = create_pipeline()

    # Example searches
    queries = ["terrier", "retriever", "small"]
    for query in queries:
        print(f"\n--- Searching for: '{query}' ---")
        results = pipeline.search(query, top_k=3)
        for result in results:
            print(f"✓ {result.content[:100]}")
