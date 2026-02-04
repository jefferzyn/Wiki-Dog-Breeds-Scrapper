"""
Main entry point for the Dog Breed Scraper and Search System.
"""

from pipeline import create_pipeline


def main():
    """Initialize and run the dog breed search system."""
    print("🐕 Dog Breed Wiki Scraper & Search System\n")

    # Create and initialize the pipeline
    pipeline = create_pipeline()

    # Example searches
    search_queries = [
        "terrier",
        "retriever",
        "german",
        "shepherd",
        "hound",
    ]

    print("\n" + "=" * 50)
    print("SEARCH RESULTS")
    print("=" * 50)

    for query in search_queries:
        print(f"\n🔍 Query: '{query}'")
        results = pipeline.search(query, top_k=3)

        if results:
            for i, result in enumerate(results, 1):
                breed_name = result.meta.get("breed_name", "Unknown")
                url = result.meta.get("url", "")
                print(f"  {i}. {breed_name}")
                if url:
                    print(f"     {url}")
        else:
            print("  No results found")


if __name__ == "__main__":
    main()
