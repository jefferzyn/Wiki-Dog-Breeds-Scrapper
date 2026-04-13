#!/usr/bin/env python
"""
Quick test script to verify the RAG pipeline works without errors.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend_api import QABackend

print("=" * 60)
print("Testing RAG Pipeline Backend")
print("=" * 60)

# Initialize backend
print("\n1. Initializing backend...")
backend = QABackend(use_openai=False, use_hf=False)

print("2. Checking status...")
status = backend.get_status()
print(f"   Initialized: {status['initialized']}")
print(f"   Status: {status['status']}")

# Initialize with a small limit for quick test
print("\n3. Loading and indexing documents (limit 5 for quick test)...")
try:
    init_result = backend.initialize(limit=5)
    print(f"   ✅ {init_result['status']}")
    print(f"   URLs indexed: {init_result['urls_indexed']}")
    print(f"   Documents created: {init_result['documents']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test a simple question
print("\n4. Testing question answering...")
try:
    question = "What are dog breeds?"
    print(f"   Question: '{question}'")
    
    response = backend.answer_question(question, save_to_log=False)
    print(f"   ✅ Success!")
    print(f"   Confident: {response.is_confident}")
    print(f"   Answer length: {len(response.answer)} chars")
    print(f"   Documents retrieved: {len(response.retrieved_docs) if response.retrieved_docs else 0}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! System is working correctly.")
print("=" * 60)
print("\nNow you can run: streamlit run streamlit_app.py")
