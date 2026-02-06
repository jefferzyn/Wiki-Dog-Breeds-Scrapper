"""
Main entry point for the Dog Breed Scraper and Search System.
"""

from pipeline import create_pipeline


def show_menu():
    """Display the main menu."""
    print("\n" + "=" * 50)
    print("DOG BREED WIKI SEARCH")
    print("=" * 50)
    print("1. List all dog breeds")
    print("2. Search for a breed")
    print("3. Get breed info by name")
    print("4. Exit")
    print("-" * 50)


def list_all_breeds(pipeline):
    """List all dog breeds."""
    all_docs = pipeline.document_store.filter_documents()
    print(f"\n{'=' * 50}")
    print(f"ALL DOG BREEDS ({len(all_docs)} breeds)")
    print("=" * 50)
    
    for i, doc in enumerate(all_docs, 1):
        breed_name = doc.meta.get("breed_name", "Unknown")
        print(f"{i:3}. {breed_name}")
    
    input("\nPress Enter to continue...")


def search_breeds(pipeline):
    """Search for dog breeds."""
    query = input("\nEnter search term: ").strip()
    if not query:
        print("No search term entered.")
        return
    
    results = pipeline.search(query, top_k=10)
    
    if results:
        print(f"\nFound {len(results)} results for '{query}':")
        print("-" * 40)
        for i, result in enumerate(results, 1):
            breed_name = result.meta.get("breed_name", "Unknown")
            url = result.meta.get("url", "")
            print(f"{i}. {breed_name}")
            print(f"   {url}")
    else:
        print(f"No results found for '{query}'")
    
    input("\nPress Enter to continue...")


def get_breed_info(pipeline):
    """Get detailed info about a specific breed."""
    name = input("\nEnter breed name: ").strip()
    if not name:
        print("No breed name entered.")
        return
    
    # Search for exact or close matches
    results = pipeline.search(name, top_k=5)
    
    if results:
        # Find best match
        best_match = None
        for result in results:
            breed_name = result.meta.get("breed_name", "").lower()
            if name.lower() in breed_name or breed_name in name.lower():
                best_match = result
                break
        
        if not best_match:
            best_match = results[0]
        
        breed_name = best_match.meta.get("breed_name", "Unknown")
        url = best_match.meta.get("url", "N/A")
        content = best_match.content
        
        print(f"\n{'=' * 50}")
        print(f"BREED: {breed_name}")
        print("=" * 50)
        print(f"URL: {url}")
        print(f"\nDescription: {content[:300]}..." if len(content) > 300 else f"\nDescription: {content}")
    else:
        print(f"No breed found matching '{name}'")
    
    input("\nPress Enter to continue...")


def main():
    """Initialize and run the dog breed search system."""
    print("\n" + "=" * 50)
    print("Dog Breed Wiki Scraper & Search System")
    print("=" * 50)
    print("\nInitializing...")

    # Create and initialize the pipeline
    pipeline = create_pipeline()
    
    all_docs = pipeline.document_store.filter_documents()
    print(f"Loaded {len(all_docs)} dog breeds!")

    while True:
        show_menu()
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            list_all_breeds(pipeline)
        elif choice == "2":
            search_breeds(pipeline)
        elif choice == "3":
            get_breed_info(pipeline)
        elif choice == "4":
            print("\nBye!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
