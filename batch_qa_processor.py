"""
Batch Q&A Processor for Dog Breed QA System

Reads questions from a file and processes them through the RAG pipeline,
saving all questions and answers to the log file.

Usage:
    python batch_qa_processor.py                              # Process all questions
    python batch_qa_processor.py --file custom_questions.txt  # Use custom file
    python batch_qa_processor.py --limit 10                   # Process first 10 questions
    python batch_qa_processor.py --use-openai                 # Use OpenAI
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List

# Import the QA system
from qa_program import DogBreedQA


def load_questions(file_path: str) -> List[str]:
    """Load questions from a text file.
    
    Args:
        file_path: Path to file containing questions (one per line, numbered)
        
    Returns:
        List of question strings
    """
    questions = []
    
    if not os.path.exists(file_path):
        print(f"Error: Question file not found: {file_path}")
        return questions
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Remove numbering if present (e.g., "1. Question" -> "Question")
            if line[0].isdigit() and '. ' in line:
                question = line.split('. ', 1)[1]
            else:
                question = line
            
            if question:
                questions.append(question)
    
    return questions


def process_batch_qa(qa_system: DogBreedQA, questions: List[str], 
                     batch_delay: float = 2.0) -> None:
    """Process a batch of questions through the QA system.
    
    Args:
        qa_system: Initialized DogBreedQA system
        questions: List of questions to process
        batch_delay: Delay between questions (seconds) to avoid overwhelming system
    """
    print(f"\n{'=' * 80}")
    print(f"BATCH Q&A PROCESSING")
    print(f"{'=' * 80}")
    print(f"Processing {len(questions)} questions...")
    print(f"Log file: {qa_system.get_log_file_path()}\n")
    
    successful = 0
    failed = 0
    
    for i, question in enumerate(questions, 1):
        try:
            print(f"[{i}/{len(questions)}] Processing: {question[:60]}...", end=" ", flush=True)
            
            # Get answer from QA system
            answer = qa_system.get_answer(question)
            
            # Save to log
            qa_system.save_qa_pair(question, answer, "Batch Processing")
            
            successful += 1
            print("✓")
            
            # Delay between requests to be polite to system
            if i < len(questions):
                time.sleep(batch_delay)
                
        except Exception as e:
            failed += 1
            print(f"✗ Error: {str(e)[:50]}")
    
    # Print summary
    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Successfully processed: {successful}/{len(questions)}")
    print(f"Failed: {failed}")
    print(f"\nAll results saved to: {qa_system.get_log_file_path()}")
    print(f"{'=' * 80}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch process dog breed questions through the QA system"
    )
    parser.add_argument(
        "--file", "-f", type=str, default="data/dog_breed_questions.txt",
        help="Path to question file (default: data/dog_breed_questions.txt)"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=0,
        help="Limit number of questions to process (0 = all)"
    )
    parser.add_argument(
        "--urls-limit", "-ul", type=int, default=0,
        help="Limit number of URLs to index (0 = all)"
    )
    parser.add_argument(
        "--use-openai", action="store_true",
        help="Use OpenAI for answer generation"
    )
    parser.add_argument(
        "--use-hf", action="store_true",
        help="Use HuggingFace API for answer generation"
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=2.0,
        help="Delay between questions in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--urls-dir", type=str, default="data/urls",
        help="Directory containing URL files"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="data/qa_outputs",
        help="Output directory for Q&A logs"
    )
    return parser.parse_args()


def main():
    """Main entry point for batch processor."""
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("DOG BREED QA SYSTEM - BATCH PROCESSOR")
    print("=" * 80)
    
    # Load questions
    print(f"\nLoading questions from: {args.file}")
    questions = load_questions(args.file)
    
    if not questions:
        print("No questions found!")
        return
    
    print(f"Loaded {len(questions)} questions")
    
    # Apply limit if specified
    if args.limit > 0:
        questions = questions[:args.limit]
        print(f"Limited to {args.limit} questions")
    
    # Initialize QA system
    print("\nInitializing QA system...")
    use_openai = args.use_openai or bool(os.getenv("OPENAI_API_KEY"))
    use_hf = args.use_hf or bool(os.getenv("HF_TOKEN"))
    
    qa = DogBreedQA(
        urls_dir=args.urls_dir,
        use_openai=use_openai,
        use_hf=use_hf,
        output_dir=args.output_dir
    )
    
    # Load and index URLs
    print(f"Loading URLs from: {args.urls_dir}")
    url_data = qa.load_urls()
    
    if args.urls_limit > 0:
        url_data = url_data[:args.urls_limit]
        print(f"Limited to {args.urls_limit} URLs")
    
    # Show indexing choice if needed
    if len(url_data) > 20:
        print(f"\nFound {len(url_data)} URLs. Indexing all for batch processing...")
        print("(This may take 10-20 minutes)")
    
    # Initialize system
    qa.initialize(url_data)
    
    # Process batch
    start_time = time.time()
    process_batch_qa(qa, questions, batch_delay=args.delay)
    elapsed_time = time.time() - start_time
    
    # Print timing statistics
    print(f"Batch processing took: {elapsed_time:.1f} seconds")
    print(f"Average time per question: {elapsed_time/len(questions):.1f} seconds")
    print(f"\nTo review your results, open: {qa.get_log_file_path()}")


if __name__ == "__main__":
    main()
