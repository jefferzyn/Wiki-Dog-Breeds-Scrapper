#!/usr/bin/env python3
"""
Diagnostic script to test the improved QA system.

Helps verify that:
1. Document retrieval is working (MRR > 0.0)
2. Hallucination prevention is active
3. References are properly tracked
4. Sources are being cited correctly
"""

import json
from qa_program import DogBreedQA


def test_basic_retrieval():
    """Test if the retriever can find any documents."""
    print("\n" + "="*70)
    print("TEST 1: Basic Retrieval Functionality")
    print("="*70)
    
    qa = DogBreedQA()
    qa.initialize()
    
    # Simple breed question that should have results
    question = "What are the main characteristics of a Labrador Retriever?"
    print(f"\nQuestion: {question}")
    
    result = qa.ask(question)
    
    # Check retriever output
    if "retriever" in result and "documents" in result["retriever"]:
        docs = result["retriever"]["documents"]
        print(f"✓ Retriever found {len(docs)} documents")
        
        if len(docs) > 0:
            print(f"  Top 3 documents:")
            for i, doc in enumerate(docs[:3], 1):
                breed = doc.meta.get('breed_name', 'Unknown') if doc.meta else 'Unknown'
                url = doc.meta.get('url', 'Unknown') if doc.meta else 'Unknown'
                print(f"    {i}. {breed}")
                print(f"       URL: {url}")
                print(f"       Content preview: {doc.content[:80]}...")
        else:
            print("✗ No documents found!")
    else:
        print("✗ Retriever output not found!")
    
    return result


def test_hallucination_prevention():
    """Test if system prevents hallucination on unanswerable questions."""
    print("\n" + "="*70)
    print("TEST 2: Hallucination Prevention")
    print("="*70)
    
    qa = DogBreedQA(use_openai=False)  # Test without LLM first
    qa.initialize()
    
    # Question unlikely to be in Wikipedia
    question = "How much does a custom dog house cost in 2026?"
    print(f"\nQuestion (not in Wikipedia): {question}")
    
    result = qa.get_answer_with_references(question)
    
    print(f"Retrieval Success: {result['retrieval_success']}")
    print(f"Number of Sources: {result['num_sources']}")
    print(f"Answer preview: {result['answer'][:150]}...")
    
    if not result['retrieval_success'] and result['num_sources'] == 0:
        print("✓ Correctly rejected question with no sources!")
    elif result['retrieval_success']:
        print("⚠ Found sources - this is OK if Wikipedia has relevant info")
    

def test_source_attribution():
    """Test that answers properly cite sources."""
    print("\n" + "="*70)
    print("TEST 3: Source Attribution & References")
    print("="*70)
    
    qa = DogBreedQA()
    qa.initialize()
    
    question = "Which dog breeds are considered hypoallergenic?"
    print(f"\nQuestion: {question}")
    
    result = qa.get_answer_with_references(question)
    
    print(f"\nRetrieval Status: {'SUCCESS' if result['retrieval_success'] else 'FAILED'}")
    print(f"Number of Sources Used: {result['num_sources']}")
    
    if result['num_sources'] > 0:
        print(f"\nReferences Retrieved:")
        for ref in result['references']:
            print(f"\n  {ref['source_id']} - {ref['breed']}")
            print(f"  URL: {ref['url']}")
            print(f"  Snippet: {ref['snippet'][:120]}...")
    else:
        print("\n✗ No sources found!")
    
    # Check if answer cites sources (if LLM is used)
    if "[Source" in result['answer']:
        print("\n✓ Answer includes citations [Source X]!")
    elif result['num_sources'] > 0:
        print("\n⚠ Sources found but answer may not cite them (check LLM config)")


def test_document_statistics():
    """Show statistics about indexed documents."""
    print("\n" + "="*70)
    print("TEST 4: Document Index Statistics")
    print("="*70)
    
    qa = DogBreedQA()
    qa.initialize()
    
    doc_count = qa.document_store.count_documents()
    print(f"\nTotal documents in index: {doc_count}")
    
    # Sample some documents to check metadata
    print(f"\nSampling first 3 documents for metadata check:")
    
    try:
        # Get sample docs (limit to 3)
        docs = qa.document_store.filter_documents(
            filters={"field": "meta", "operator": "!=", "value": None}
        )[:3]
        
        for i, doc in enumerate(docs, 1):
            if doc.meta:
                print(f"\n  Document {i}:")
                print(f"    Content length: {len(doc.content)} characters")
                print(f"    Breed: {doc.meta.get('breed_name', 'N/A')}")
                print(f"    URL: {doc.meta.get('url', 'N/A')[:60]}...")
                print(f"    Has embedding: {'embedding' in doc and doc.embedding is not None}")
            else:
                print(f"\n  Document {i}: No metadata found")
    except Exception as e:
        print(f"  Note: Could not retrieve sample docs: {e}")


def test_benchmark_questions():
    """Test a variety of questions to see success rate."""
    print("\n" + "="*70)
    print("TEST 5: Benchmark Questions (Success Rate)")
    print("="*70)
    
    qa = DogBreedQA()
    qa.initialize()
    
    test_questions = [
        "What are the main characteristics of a Labrador Retriever?",
        "Which dog breeds are considered hypoallergenic?",
        "What is the average lifespan of a German Shepherd?",
        "Which breeds are best for apartment living?",
        "What are the grooming needs of a Poodle?",
    ]
    
    successful = 0
    failed = 0
    
    print(f"\nTesting {len(test_questions)} questions...")
    
    for i, question in enumerate(test_questions, 1):
        result = qa.get_answer_with_references(question)
        
        if result['retrieval_success'] and result['num_sources'] > 0:
            print(f"  {i}. ✓ {question[:50]}... ({result['num_sources']} sources)")
            successful += 1
        else:
            print(f"  {i}. ✗ {question[:50]}... (no sources)")
            failed += 1
    
    success_rate = (successful / len(test_questions)) * 100
    print(f"\nSuccess Rate: {successful}/{len(test_questions)} ({success_rate:.1f}%)")
    
    if success_rate > 80:
        print("✓ Excellent! Retrieval system is working well!")
    elif success_rate > 50:
        print("⚠ Moderate performance. Consider re-indexing with fresh data.")
    else:
        print("✗ Low success rate. Check document indexing and embeddings.")


def main():
    """Run all diagnostic tests."""
    print("\n" + "#"*70)
    print("# Dog Breed QA System - Diagnostic Test Suite")
    print("#"*70)
    
    try:
        test_basic_retrieval()
        test_document_statistics()
        test_hallucination_prevention()
        test_source_attribution()
        test_benchmark_questions()
        
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        print("""
If you see:
  ✓ marks - the system is working correctly!
  ✗ marks - you may need to re-index: python qa_program.py --index-only
  ⚠ marks - system is working but can be improved

Key improvements made:
  • Larger semantic chunks (500 words) instead of 200
  • Better metadata preservation (breed names, URLs)
  • Hallucination prevention (rejects answers without sources)
  • Improved reference tracking and source attribution

Next steps:
  1. Re-index if you see ✗ marks:
     python qa_program.py --index-only
  
  2. Run evaluation to get metrics:
     python qa_program.py --eval --use-openai
  
  3. Try the interactive mode:
     python qa_program.py --use-openai
        """)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
