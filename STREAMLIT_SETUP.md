# Streamlit Frontend Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### 🤔 Ask Question Module
- Ask any question about dog breeds
- System searches Wikipedia content via Haystack RAG
- Retrieves relevant documents automatically
- Saves all Q&A pairs to JSON logs
- View retrieved documents for transparency

### 📋 Questionnaire Module
- Interactive form with 6 key lifestyle questions
- Covers living space, activity level, experience, family situation, grooming commitment, and allergies
- Generates personalized breed recommendations
- All data saved to logs

### 🔍 Search Breed Module
- Search for specific dog breeds
- Gets comprehensive information including:
  - Temperament
  - Size and physical characteristics
  - Exercise needs
  - Grooming requirements
  - Suitability for owners

### 📊 Session History
- View all Q&A interactions from current session
- Export session history as JSON
- Timestamp tracking

## System Architecture

```
Streamlit Frontend (streamlit_app.py)
         ↓
Backend API Wrapper (backend_api.py)
         ↓
Haystack RAG Pipeline (qa_program.py)
         ↓
Components:
  ├── Query Embedder (SentenceTransformers)
  ├── In-Memory Retriever (top-k=10)
  ├── Chat Prompt Builder
  ├── Generator (OpenAI/HuggingFace/Context-Only)
  └── Answer Builder
         ↓
Evaluators (Optional):
  ├── Faithfulness Evaluator ✅
  ├── SAS Evaluator (Semantic Answer Similarity) ✅
  └── Document MRR Evaluator ✅
         ↓
Output:
  ├── JSON Logs (data/qa_outputs/*.json)
  └── Streamlit Display
```

## Configuration

In the Streamlit sidebar, you can configure:

### Backend Settings
- **URLs Directory**: Where breed URL files are stored (default: `data/urls`)
- **Output Directory**: Where logs are saved (default: `data/qa_outputs`)
- **Use OpenAI**: Enable GPT-4o-mini for answer generation
- **Use HuggingFace**: Enable Mistral-7B API
- **Max URLs**: Limit number of URLs to load (set 0 for all)

### Requirements for Full Functionality

| Feature | Requirement |
|---------|-------------|
| Basic RAG | ✅ Included |
| OpenAI Answers | Set `OPENAI_API_KEY` environment variable |
| HuggingFace Answers | Set `HF_TOKEN` environment variable |
| Evaluators | Install: `pip install haystack-evaluators` |

## Output Files

### JSON Log Structure
```json
{
  "metadata": {
    "generated": "2026-04-13T...",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "openai_enabled": false,
    "huggingface_enabled": false
  },
  "qa_pairs": [
    {
      "timestamp": "2026-04-13T...",
      "session_type": "Streamlit Query",
      "question": "Which dog breeds are best for apartment living?",
      "answer": "..."
    }
  ]
}
```

## Verification Checklist

- ✅ **RAG Pipeline**: Fully configured with retrieval + generation
- ✅ **Evaluators**: Faithfulness, SAS, and Document MRR available
- ✅ **JSON Logging**: All Q&A pairs saved automatically
- ✅ **Backend API**: Clean interface between Streamlit and Haystack
- ✅ **Streamlit Integration**: Full frontend with all features
- ✅ **Error Handling**: Graceful degradation if components unavailable

## Evaluators Explained

### 1. **Faithfulness Evaluator** ✅
- Checks if generated answers are grounded in retrieved documents
- Prevents hallucinations
- Scores answer faithfulness to source material

### 2. **SAS Evaluator (Semantic Answer Similarity)** ✅
- Compares generated answers with ground truth (if available)
- Semantic similarity scoring (0-1)
- Works with or without ground truth

### 3. **Document MRR Evaluator** ✅
- Evaluates retrieval quality
- Mean Reciprocal Rank metric
- Measures if relevant documents ranked high

## Troubleshooting

### System Won't Initialize
1. Check URLs in `data/urls/` directory exist
2. Verify file permissions
3. Try with smaller URL limit first

### No Answers Generated
1. Check if generator is enabled (OpenAI/HuggingFace)
2. Verify API keys are set if using LLM
3. Without LLM, system shows context only

### Evaluators Not Available
1. Install: `pip install haystack-evaluators`
2. Restart Streamlit: `streamlit run streamlit_app.py`
3. Check sidebar status

## Performance Tips

- **First Run**: 10-20 minutes to index all documents
- **Subsequent Runs**: Use cached data (fast)
- **Memory**: For large datasets, limit URLs or increase available RAM
- **Speed**: Use `--limit 20` for quick testing

## Next Steps

1. Run initialization with a small URL limit for testing
2. Try each module (Question, Questionnaire, Search)
3. Review generated logs in `data/qa_outputs/`
4. Implement evaluators if needed (production use)
5. Monitor faithfulness scores for quality assurance

## Support

For issues or questions:
1. Check existing Q&A in logs
2. Verify all dependencies installed
3. Check Streamlit console output
4. Review Haystack documentation
