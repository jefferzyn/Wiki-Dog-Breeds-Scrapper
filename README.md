# Wiki-Dog-Breeds-Scrapper

A comprehensive Python tool for scraping, searching, and answering questions about dog breeds from Wikipedia. This project provides multiple interfaces for interacting with dog breed data using web scraping, semantic search, and QA capabilities.

## Features

### Core Functionality
- **Web Scraping**: Scrapes 600+ dog breeds from Wikipedia's List of Dog Breeds
- **Semantic Search**: Uses Haystack and BM25 for fast, intelligent breed searching
- **Question Answering**: RAG-based QA system for dog breed queries
- **Batch Processing**: Process multiple questions at once and generate logs
- **Multiple Interfaces**: Interactive CLI, search pipeline, and QA system

### Advanced Features
- Wikipedia page description fetching for dog breeds
- Customizable embedding models (Sentence Transformers)
- Batch Q&A processing with logging
- Optional OpenAI/HuggingFace integration for advanced QA
- Document retrieval evaluation metrics

## Project Structure

```
.
├── main.py                      # Interactive dog breed search CLI
├── scraper.py                   # Wikipedia dog breed web scraper
├── pipeline.py                  # Haystack search pipeline
├── qa_program.py                # RAG-based Q&A system
├── batch_qa_processor.py        # Batch question processing
├── requirements.txt             # Python dependencies
├── data/
│   ├── dog_breeds.json         # Cached breed data
│   ├── dog_breed_questions.txt  # Sample questions for QA
│   ├── urls/                    # Individual breed Wikipedia links
│   └── qa_outputs/              # QA logs and results
└── docs/                        # Documentation files
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Wiki-Dog-Breeds-Scrapper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Interactive Search (main.py)

The primary CLI interface for browsing and searching dog breeds:

```bash
python main.py
```

**Menu Options:**
- **Option 1**: List all 600+ dog breeds with numbering
- **Option 2**: Search for breeds by keyword (e.g., "terrier", "shepherd", "small")
- **Option 3**: Get detailed information about a specific breed by name
- **Option 4**: Exit the application

**Example:**
```
DOG BREED WIKI SEARCH
==================================================
1. List all dog breeds
2. Search for a breed
3. Get breed info by name
4. Exit
--------------------------------------------------
Enter your choice (1-4): 2
Enter search term: terrier

Found 10 results for 'terrier':
1. Airedale Terrier
   https://en.wikipedia.org/wiki/Airedale_Terrier
2. American Pit Bull Terrier
   https://en.wikipedia.org/wiki/American_Pit_Bull_Terrier
...
```

### 2. Q&A System (qa_program.py)

An advanced RAG (Retrieval-Augmented Generation) system for answering questions about dog breeds:

```bash
# Interactive Q&A mode (default)
python qa_program.py

# Index documents only (no Q&A)
python qa_program.py --index-only

# Limit Wikipedia pages to process
python qa_program.py --limit 50

# Enable OpenAI for advanced answers
python qa_program.py --use-openai

# Enable HuggingFace API for answers
python qa_program.py --use-hf

# Show system information
python qa_program.py --show-info
```

**Features:**
- Fetches and indexes Wikipedia breed pages
- Uses semantic embeddings for relevant document retrieval
- Generates contextual answers based on retrieved content
- Built-in question logging with timestamps
- Optional LLM integration (OpenAI/HuggingFace)
- Supports batching for large-scale processing

### 3. Batch Question Processing (batch_qa_processor.py)

Process multiple questions from a file automatically:

```bash
# Process all questions in default file
python batch_qa_processor.py

# Use custom question file
python batch_qa_processor.py --file custom_questions.txt

# Process first N questions
python batch_qa_processor.py --limit 20

# Enable OpenAI for answers
python batch_qa_processor.py --use-openai

# Enable HuggingFace API
python batch_qa_processor.py --use-hf
```

**Input Format:**
Questions should be in `data/dog_breed_questions.txt`:
```
1. What dog breeds are good for apartments?
2. Which breeds are hypoallergenic?
3. Tell me about German Shepherds
4. What small dog breeds are good with children?
```

**Output:**
Results are saved to `data/qa_outputs/qa_log_[timestamp].json` with:
- Timestamp
- Question asked
- Text answer
- Retrieved documents
- Processing statistics

### 4. Programmatic Usage

#### Using the Search Pipeline

```python
from pipeline import create_pipeline

# Create and initialize pipeline
pipeline = create_pipeline(fetch_descriptions=False)

# Search for breeds
results = pipeline.search("terrier", top_k=10)

for result in results:
    breed_name = result.meta.get("breed_name")
    url = result.meta.get("url")
    print(f"{breed_name}: {url}")
```

#### Using the Scraper Directly

```python
from scraper import DogBreedScraper

scraper = DogBreedScraper()

# Scrape breed list with descriptions
breeds = scraper.scrape(fetch_descriptions=True)

for breed in breeds:
    print(f"Breed: {breed['name']}")
    print(f"URL: {breed['url']}")
    print(f"Description: {breed['description']}\n")
```

#### Using the QA System

```python
from qa_program import DogBreedQA

# Initialize QA system
qa = DogBreedQA()
qa.index_documents()

# Ask a question
answer = qa.answer_question("What dog breeds are good for families?")
print(answer)

# Get answer with sources
answer, sources = qa.answer_with_sources("Tell me about Golden Retrievers")
print(f"Answer: {answer}")
print(f"Sources: {sources}")
```

## Module Documentation

### scraper.py

**Main Class:** `DogBreedScraper`

Scrapes dog breed information from Wikipedia's List of Dog Breeds page.

**Key Methods:**
- `fetch_page(url)` - Fetch HTML content from URL
- `parse_breeds(html)` - Extract breed information from HTML
- `fetch_breed_description(url)` - Get breed description from Wikipedia page
- `scrape(fetch_descriptions=False)` - Main scraping method
- `save_breed_urls(breeds, output_dir)` - Save breed URLs to files
- `clean_breed_name(name)` - Clean breed names (remove citations)

**Returns:** List of dictionaries with keys: `name`, `url`, `description`

### pipeline.py

**Main Class:** `DogBreedPipeline`

Manages Haystack search pipeline for dog breed retrieval.

**Key Methods:**
- `load_data(fetch_descriptions=False)` - Load scraped breed data
- `build_pipeline()` - Build Haystack search pipeline
- `search(query, top_k=5)` - Search for breeds matching query
- `initialize(fetch_descriptions=False)` - Set up entire pipeline

**Components:**
- InMemoryDocumentStore for document storage
- InMemoryBM25Retriever for keyword matching
- Haystack Pipeline orchestration

### qa_program.py

**Main Class:** `DogBreedQA`

Advanced RAG system for question answering about dog breeds.

**Key Methods:**
- `index_documents()` - Fetch and index Wikipedia breed pages
- `answer_question(question)` - Get answer to a question
- `answer_with_sources(question)` - Get answer with source documents
- `get_pipeline_status()` - Print system configuration

**Features:**
- Custom Wikipedia fetcher with proper headers and delays
- Document cleaning and splitting
- Sentence Transformer embeddings
- BM25 + semantic retrieval
- Optional LLM integrations
- Comprehensive logging

### batch_qa_processor.py

**Main Functions:**

- `load_questions(file_path)` - Load questions from file
- `process_batch_qa(qa_system, questions, batch_delay)` - Process question batch
- `save_results(results, log_file)` - Save results to log file

**Handles:**
- Question numbering parsing
- Error recovery
- Rate limiting between questions
- Comprehensive result logging

## Dependencies

- `requests` - HTTP requests for web scraping
- `beautifulsoup4` - HTML parsing
- `haystack-ai` - RAG and search pipeline framework
- `sentence-transformers` - Document embeddings
- `trafilatura` - Content extraction from web pages
- `lxml` - XML/HTML processing
- `openai` - OpenAI API integration (optional)
- `nltk` - Natural language processing (optional)

Install all at once:
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

You can set optional environment variables for advanced features:

```bash
# Enable OpenAI (requires API key)
export OPENAI_API_KEY="your-key-here"

# HuggingFace API token
export HF_API_KEY="your-hf-token"

# Custom embedding model
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

### Customization

Modify parameters in each module:

**qa_program.py:**
- `EMBEDDING_MODEL` - Change embedding model
- `TIMEOUT` - Adjust Wikipedia fetch timeout
- `DELAY` - Control request rate limiting

**batch_qa_processor.py:**
- `batch_delay` - Delay between questions
- `QUESTION_FILE` - Default questions file path

**pipeline.py:**
- `top_k` - Number of results to return
- Haystack pipeline components

## Troubleshooting

### Module Not Found Errors
**Solution:** Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### No Results in Searches
**Solution:** Verify data was loaded properly
```bash
python qa_program.py --show-info
```

### Slow Performance
**Solution:** Reduce `--limit` or disable description fetching
```bash
python qa_program.py --limit 100 --index-only
```

### Network Errors
**Solution:** Check internet connection and Wikipedia accessibility
```bash
pip install --upgrade requests urllib3
```

## Output Files

- `data/dog_breeds.json` - Cached breed list
- `data/qa_outputs/qa_log_*.json` - Q&A session logs with timestamps
- `data/urls/*.txt` - Individual breed URL files

## Examples

### Find Small Dog Breeds
```bash
python main.py
# Choose option 2 (Search)
# Enter: small
```

### Answer Questions About Dog Breeds
```bash
python qa_program.py
# Enter: "What dog breeds are hypoallergenic?"
```

### Batch Process Questions
```bash
# Create data/dog_breed_questions.txt with your questions
python batch_qa_processor.py --limit 50
# Check data/qa_outputs/qa_log_*.json for results
```

## Performance Notes

- First run downloads and indexes Wikipedia pages (5-15 minutes depending on connection)
- Subsequent runs use cached data (fast)
- BM25 search is optimized for keyword matching
- Embedding-based retrieval provides semantic understanding
- Batch processing adds 1-2 seconds per question

## Limitations

- Requires active internet connection for Wikipedia scraping
- Rate limited to be polite to Wikipedia servers
- Embedding model requires sufficient RAM
- OpenAI/HuggingFace features require API credentials

## Future Enhancements

- PostgreSQL/SQLite database backend
- Async processing for faster batch operations
- Web UI interface
- More LLM provider support
- Document caching strategies
- Evaluation metrics dashboard

## License

[Add your license here]

## Support

For issues or questions, please check:
1. README.md (this file)
2. QUICK_REFERENCE.md (common commands)
3. QA_LOGGING_GUIDE.md (Q&A system help)
4. BATCH_PROCESSING_GUIDE.md (batch operations)

## Contributing

Contributions welcome! Please:
1. Test your changes
2. Update documentation
3. Follow existing code style
4. Submit pull requests with descriptions
