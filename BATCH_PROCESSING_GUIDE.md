# Batch Q&A Processing Guide

## Overview
The batch Q&A processor allows you to automatically process multiple dog breed questions through your RAG system, generating comprehensive Q&A datasets for training, evaluation, and reference.

## Files

### `data/dog_breed_questions.txt`
Contains 50 pre-formulated questions covering:
- Breed characteristics and temperament
- Health issues and lifespan
- Exercise and grooming needs
- Origin and history
- Size and coat type
- Suitability for different owners
- Popular and rare breeds
- Training difficulty
- And more...

### `batch_qa_processor.py`
Python script that:
- Reads questions from a file
- Processes each through the RAG pipeline
- Saves all Q&A pairs to timestamped log files
- Provides statistics and progress tracking

## Quick Start

### Basic Usage (Process All Questions)
```bash
python batch_qa_processor.py
```

This will:
1. Load all 50 questions from `data/dog_breed_questions.txt`
2. Index your dog breed documents (may take 10-20 minutes)
3. Process each question through the RAG pipeline
4. Save all Q&A pairs to `data/qa_outputs/qa_log_YYYYMMDD_HHMMSS.json`

### Process First N Questions
```bash
python batch_qa_processor.py --limit 10
```

Process only the first 10 questions (faster for testing).

### Use Custom Question File
```bash
python batch_qa_processor.py --file my_questions.txt
```

Your question file should have one question per line, with optional numbering:
```
1. What are the main characteristics of a Labrador Retriever?
2. Which dog breeds require the most exercise?
...
```

### Adjust Delay Between Questions
```bash
python batch_qa_processor.py --delay 1.0
```

Set delay between questions in seconds (default: 2.0).

### Process Fewer URLs for Testing
```bash
python batch_qa_processor.py --urls-limit 20 --limit 10
```

Process only 20 URLs and 10 questions (fast demo).

### Use OpenAI
```bash
python batch_qa_processor.py --use-openai
```

Requires `OPENAI_API_KEY` environment variable.

## Complete Command Examples

### Quick Test (5 questions, 10 URLs)
```bash
python batch_qa_processor.py --limit 5 --urls-limit 10
```

**Time**: ~2-5 minutes  
**Output**: 5 Q&A pairs saved

### Medium Batch (25 questions, all URLs)
```bash
python batch_qa_processor.py --limit 25
```

**Time**: ~15-30 minutes  
**Output**: 25 Q&A pairs saved

### Full Dataset (all 50 questions)
```bash
python batch_qa_processor.py
```

**Time**: ~30-60 minutes  
**Output**: 50 comprehensive Q&A pairs

### With OpenAI and Custom Timing
```bash
python batch_qa_processor.py --use-openai --delay 3.0 --limit 30
```

**Time**: ~2-3 minutes per question  
**Total**: ~90 minutes  
**Output**: 30 Q&A pairs with generated answers

## Output Format

All Q&A pairs are saved in this format:

```
[2026-04-06 15:30:42] Batch Processing
────────────────────────────────────────────────────────────────────────────────
QUESTION:
What are the main characteristics of a Labrador Retriever?

ANSWER:
[Retrieved Context - No LLM configured]

Labrador Retrievers are one of the most popular dog breeds worldwide, known for...

================================================================================
```

Each log file contains:
- **Session header** with metadata
- **Timestamp** for each question
- **Original question**
- **Full answer** with context
- **Clear separators** for easy parsing

## Processing Statistics

The batch processor provides:
- **Progress tracking**: `[5/50] Processing...`
- **Success/failure count**: `Successfully processed: 50/50`
- **Time statistics**: Average time per question
- **Log file location**: Easy reference to saved file

Example output:
```
================================================================================
BATCH PROCESSING COMPLETE
================================================================================
Successfully processed: 50/50
Failed: 0

Average time per question: 12.3 seconds

All results saved to: data/qa_outputs/qa_log_20260406_153042.json
================================================================================
```

## Using Your Generated Q&A Dataset

### 1. Review Answers
```bash
# Open and read the JSON log file
type data/qa_outputs/qa_log_20260406_153042.json

# Or view in your editor
code data/qa_outputs/qa_log_20260406_153042.json

# Pretty print the JSON
powershell -Command "Get-Content data/qa_outputs/qa_log_*.json | ConvertFrom-Json | ConvertTo-Json -Depth 10"
```

### 2. Extract Questions Only
```bash
# Find all questions in the JSON log using jq (if installed)
jq '.qa_pairs[] | .question' data/qa_outputs/qa_log_*.json

# Or using Python
python -c "import json; data = json.load(open('data/qa_outputs/qa_log_20260406_153042.json')); [print(pair['question']) for pair in data['qa_pairs']]"
```

### 3. Extract Answers Only
```bash
# Find all answers in the JSON log
jq '.qa_pairs[] | .answer' data/qa_outputs/qa_log_*.json

# Or using Python
python -c "import json; data = json.load(open('data/qa_outputs/qa_log_20260406_153042.json')); [print(pair['answer']) for pair in data['qa_pairs']]"
```

### 4. Count Interactions
```bash
# See how many Q&A pairs in a file
jq '.qa_pairs | length' data/qa_outputs/qa_log_*.json

# Or using Python
python -c "import json; data = json.load(open('data/qa_outputs/qa_log_20260406_153042.json')); print(len(data['qa_pairs']))"
```

### 5. Search for Specific Topics
```bash
# Find all questions containing "breed"
jq '.qa_pairs[] | select(.question | contains("breed"))' data/qa_outputs/qa_log_*.json

# Find all questions about health
jq '.qa_pairs[] | select(.question | test("health|disease|issue"; "i"))' data/qa_outputs/qa_log_*.json

# Find all grooming-related Q&A
jq '.qa_pairs[] | select(.question | test("groom"; "i"))' data/qa_outputs/qa_log_*.json
```

## Use Cases

### 1. **System Evaluation**
- Compare outputs with/without LLM
- Test different embedding models
- Validate retrieval quality
- Assess answer consistency

### 2. **Training Data**
- Extract Q&A pairs for fine-tuning
- Create benchmark datasets
- Build evaluation metrics
- Generate synthetic training data

### 3. **Documentation**
- Create breed reference guide
- Build FAQ database
- Generate knowledge base
- Create user guides

### 4. **Quality Assurance**
- Ensure consistent answers
- Identify missing information
- Spot retrieval errors
- Track improvements

### 5. **Performance Analysis**
- Measure response times
- Track success rates
- Monitor resource usage
- Optimize batch processing

## Advanced Usage

### Process Questions with Different URLs
```bash
# Use different URL directory
python batch_qa_processor.py --urls-dir data/urls --limit 10

# Save to different output location
python batch_qa_processor.py --output-dir results/batch_1
```

### Chain Multiple Batches
```bash
# Batch 1: Questions 1-25
python batch_qa_processor.py --limit 25

# Later: Questions 26-50
python batch_qa_processor.py --limit 50 --output-dir data/qa_outputs_batch2
```

### Integration with Evaluation
```bash
# Process questions, then evaluate
python batch_qa_processor.py --limit 25 --urls-limit 50
python qa_program.py --evaluate --eval-samples 25
```

## Troubleshooting

### No Questions Loading
- Check file path is correct
- Ensure file is in UTF-8 encoding
- Verify questions are one per line

### Processing Very Slow
- Increase `--delay` to avoid system overload
- Use `--urls-limit` to reduce search space
- Try `--limit` with fewer questions for testing

### Out of Memory
- Process in smaller batches
- Use `--urls-limit` to reduce documents
- Close other applications

### Answers Too Generic
- Use `--use-openai` or `--use-hf` for LLM generation
- Ensure all URLs are indexed correctly
- Check embedding model quality

## Monitoring Progress

The script shows real-time progress:
```
[1/50] Processing: What are the main characteristics... ✓
[2/50] Processing: Which dog breeds are considered... ✓
[3/50] Processing: What is the average lifespan... ✓
```

Each `✓` means one Q&A pair successfully saved!

## Performance Tips

1. **First run**: Longer due to indexing, subsequent runs reuse the index
2. **Batch size**: 25-50 questions = good balance
3. **Delay**: 2.0 seconds = default, increase if system is slow
4. **URLs**: Limiting to 50-200 breeds = faster processing
5. **LLM**: OpenAI adds ~2-3 minutes per question

## Next Steps

### After Batch Processing
1. **Review** the generated Q&A log
2. **Extract** useful pairs for your dataset
3. **Evaluate** answer quality
4. **Iterate** by adjusting system parameters
5. **Deploy** improved version with insights

### Build Upon Results
- Use Q&A pairs to train custom models
- Create FAQ documentation
- Build evaluation benchmarks
- Develop retrieval metrics
- Improve prompt engineering

## Support

For issues:
1. Check file paths are correct
2. Verify all dependencies installed
3. Ensure adequate disk space
4. Review error messages carefully
5. Start with `--limit 5` for testing

## Statistics

Expected performance:
- **Without LLM**: 5-15 seconds per question
- **With OpenAI**: 2-3 minutes per question
- **Full batch (50 Q)**: 5-60 minutes depending on config
- **Storage per Q**: ~2-5 KB per question
- **Full 50Q dataset**: ~100-250 KB

Enjoy your batch Q&A dataset! 🚀
