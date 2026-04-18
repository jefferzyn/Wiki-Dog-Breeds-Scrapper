"""
Test QA System with Known-Good Questions and Verify Answers
This tests that the system retrieves correct answers from Wikipedia sources.
"""

import json
import time
from qa_program import DogBreedQA

# Test questions with known correct information from Wikipedia
TEST_QUESTIONS = [
    {
        "question": "What is a Labrador Retriever?",
        "expected_keywords": ["breed", "retriever", "labrador", "large", "friendly"],
        "expected_breed": "Labrador Retriever",
    },
    {
        "question": "What are the characteristics of a German Shepherd?",
        "expected_keywords": ["german", "shepherd", "large", "intelligent", "loyal", "working"],
        "expected_breed": "German Shepherd",
    },
    {
        "question": "What is a Poodle?",
        "expected_keywords": ["poodle", "hypoallergenic", "intelligent", "coat"],
        "expected_breed": "Poodle",
    },
    {
        "question": "Tell me about Golden Retrievers",
        "expected_keywords": ["golden", "retriever", "friendly", "intelligent", "coat"],
        "expected_breed": "Golden Retriever",
    },
    {
        "question": "What breed is a Chihuahua?",
        "expected_keywords": ["chihuahua", "small", "toy", "alert"],
        "expected_breed": "Chihuahua",
    },
    {
        "question": "Describe a Bulldog",
        "expected_keywords": ["bulldog", "muscular", "stocky", "wrinkled", "face"],
        "expected_breed": "Bulldog",
    },
    {
        "question": "What are Husky traits?",
        "expected_keywords": ["husky", "sled", "energetic", "siberian", "arctic"],
        "expected_breed": "Siberian Husky",
    },
    {
        "question": "Tell me about Beagles",
        "expected_keywords": ["beagle", "hunting", "scent", "hound", "small"],
        "expected_breed": "Beagle",
    },
]

def test_qa_system():
    """Test the QA system with known questions."""
    
    print("\n" + "="*70)
    print("TESTING QA SYSTEM WITH KNOWN QUESTIONS")
    print("="*70)
    
    # Initialize QA system
    print("\n[1] Initializing QA system...")
    qa = DogBreedQA(use_openai=True)
    qa.initialize()
    print("✓ QA system initialized")
    
    # Run tests
    results = []
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        expected_breed = test_case["expected_breed"]
        
        print(f"\n[{i}] Testing: '{question}'")
        print("-" * 70)
        
        try:
            # Get answer with references
            result = qa.get_answer_with_references(question)
            
            # Check if retrieval succeeded
            if not result.get("retrieval_success"):
                print("✗ FAILED: No documents retrieved")
                print(f"  Answer: {result.get('answer', 'N/A')}")
                failed += 1
                results.append({
                    "question": question,
                    "status": "FAILED",
                    "reason": "No documents retrieved",
                    "answer": result.get("answer"),
                    "sources": []
                })
                continue
            
            answer = result.get("answer", "").lower()
            sources = result.get("references", [])
            num_sources = result.get("num_sources", 0)
            
            # Verify keywords present
            keywords_found = [kw for kw in expected_keywords if kw.lower() in answer]
            keyword_coverage = len(keywords_found) / len(expected_keywords)
            
            # Check sources
            has_citations = "[source" in answer.lower()
            
            # Determine pass/fail
            passed_test = (
                keyword_coverage >= 0.5 and  # At least 50% of keywords
                num_sources >= 1 and  # At least one source
                has_citations  # Has [Source X] citations
            )
            
            if passed_test:
                print(f"✓ PASSED")
                passed += 1
                status = "PASSED"
            else:
                print(f"⚠ WARNING")
                failed += 1
                status = "WARNING"
            
            # Show details
            print(f"  Keywords found: {len(keywords_found)}/{len(expected_keywords)} ({keyword_coverage*100:.0f}%)")
            print(f"  Sources retrieved: {num_sources}")
            print(f"  Has citations: {'Yes' if has_citations else 'No'}")
            print(f"  Answer preview: {answer[:150]}...")
            
            if sources:
                print(f"  Sources:")
                for src in sources[:3]:  # Show first 3 sources
                    breed = src.get("breed", "Unknown")
                    print(f"    - {breed}")
            
            results.append({
                "question": question,
                "status": status,
                "keywords_found": len(keywords_found),
                "keywords_total": len(expected_keywords),
                "sources": num_sources,
                "has_citations": has_citations,
                "answer_preview": answer[:200]
            })
            
            time.sleep(1)  # Brief pause between questions
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            failed += 1
            results.append({
                "question": question,
                "status": "ERROR",
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(TEST_QUESTIONS)}")
    print(f"Passed: {passed}")
    print(f"Failed/Warning: {failed}")
    print(f"Success rate: {passed/len(TEST_QUESTIONS)*100:.1f}%")
    
    # Detailed results
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['question']}")
        print(f"    Status: {result['status']}")
        if result['status'] == "PASSED":
            print(f"    Keywords: {result['keywords_found']}/{result['keywords_total']}")
            print(f"    Sources: {result['sources']}")
            print(f"    Citations: {result['has_citations']}")
        elif result['status'] == "ERROR":
            print(f"    Error: {result['error']}")
        else:
            print(f"    Keywords: {result.get('keywords_found', 'N/A')}/{result.get('keywords_total', 'N/A')}")
            print(f"    Sources: {result.get('sources', 'N/A')}")
    
    # Save results
    with open("data/qa_outputs/test_qa_answers.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results saved to data/qa_outputs/test_qa_answers.json")
    
    return results

if __name__ == "__main__":
    test_qa_system()
